import os
import json, os, re, time
from openai import OpenAI
import xml.etree.ElementTree as ET
from rank_bm25 import BM25Okapi
from collections import defaultdict

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL    = "qwen/qwen-2.5-72b-instruct"
CIVIL_XML = "./train2026/2026/civil.xml"
DATA_PATH = "./data/all.jsonl"
OUT_PATH  = "./results/qwen72b_structured.jsonl"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# Build retrieval
tree = ET.parse(CIVIL_XML)
articles = {}
for art in tree.findall('Article'):
    num = art.get('num')
    articles[num] = (art.find('caption').text or '') + ' ' + (art.find('text').text or '')
art_nums = list(articles.keys())

def tok_2gram(t):
    t = re.sub(r'\s+','',t)
    return [t[i:i+2] for i in range(len(t)-1)]

bm25 = BM25Okapi([tok_2gram(articles[n]) for n in art_nums])

def kanji_to_int(s):
    if not s: return None
    d={'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    r=0
    if '千' in s: p=s.split('千',1); r+=(kanji_to_int(p[0])or 1)*1000; s=p[1]
    if '百' in s: p=s.split('百',1); r+=(kanji_to_int(p[0])or 1)*100;  s=p[1]
    if '十' in s: p=s.split('十',1); t=d.get(p[0])if p[0]else 1; r+=(t if t else 1)*10; s=p[1]
    for c in s:
        if c in d: r+=d[c]
    return r or None

art_set=set(art_nums)
xref_map={}
pat=re.compile(r'第([一二三四五六七八九十百千]+)条(の[一二三四五六七八九十]+)?')
for num,text in articles.items():
    refs=set()
    for m in pat.finditer(text):
        n=kanji_to_int(m.group(1))
        if n is None: continue
        sfx=m.group(2)
        key=f"{n}-{kanji_to_int(sfx.replace('の',''))}"if sfx else str(n)
        if key in art_set: refs.add(key)
    refs.discard(num); xref_map[num]=refs

def retrieve(query,k=10):
    scores=bm25.get_scores(tok_2gram(query))
    top_idx=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:k]
    retrieved=set(art_nums[i] for i in top_idx)
    score_map={art_nums[i]:scores[i] for i in top_idx}
    for art in list(retrieved): retrieved.update(xref_map.get(art,set()))
    return retrieved,score_map

# TWO prompts — we run both and compare
STANDARD_SYSTEM = """あなたは日本の民法の専門家です。
与えられた条文と陳述文を分析し、陳述文が条文から論理的に導かれるか判断してください。
最後に必ず「判定：Y」または「判定：N」とだけ記載してください。"""

STRUCTURED_SYSTEM = """あなたは日本の民法の専門家です。
以下の手順で判断してください：

ステップ1：条文に登場する全ての法律上の当事者を列挙する（例：本人、代理人、相手方など）
ステップ2：条文の主な規定と例外・ただし書きを整理する
ステップ3：陳述文の当事者・条件・効果が条文と完全に一致するか確認する
ステップ4：一致しない点があれば何がどう違うか明示する

最後に必ず「判定：Y」または「判定：N」と記載してください。"""

FEW_SHOTS = [
    ("（補助開始の審判）\n第十五条　精神上の障害により事理を弁識する能力が不十分である者については、家庭裁判所は、本人、配偶者、四親等内の親族、後見人、後見監督人、保佐人、保佐監督人又は検察官の請求により、補助開始の審判をすることができる。\n２　本人以外の者の請求により補助開始の審判をするには、本人の同意がなければならない。",
     "本人以外の者から補助開始の申立てがされたときは、家庭裁判所は、本人の同意がなければ、補助開始の審判をすることができない。",
     "ステップ1：当事者：本人、本人以外の者、家庭裁判所\nステップ2：主規定：補助開始の審判ができる。例外：本人以外の請求には本人の同意が必要。\nステップ3：陳述文は第2項を正確に述べている。\nステップ4：相違なし。\n判定：Y"),
    ("（成年被後見人の法律行為）\n第九条　成年被後見人の法律行為は、取り消すことができる。ただし、日用品の購入その他日常生活に関する行為については、この限りでない。",
     "成年被後見人が成年後見人の同意を得ずに日用品の購入をしたときは、成年後見人は、その購入を内容とする契約を取り消すことができる。",
     "ステップ1：当事者：成年被後見人、成年後見人\nステップ2：主規定：法律行為は取消可能。ただし書き：日用品購入は取消不可。\nステップ3：陳述文は日用品購入を取消可能と主張している。\nステップ4：ただし書きと矛盾。日用品購入は取り消せない。\n判定：N"),
]

def make_messages(t1, t2, structured=True):
    system = STRUCTURED_SYSTEM if structured else STANDARD_SYSTEM
    msgs = [{"role":"system","content":system}]
    for ex_t1,ex_t2,ex_ans in FEW_SHOTS:
        msgs.append({"role":"user","content":f"【条文】\n{ex_t1}\n\n【陳述文】\n{ex_t2}"})
        msgs.append({"role":"assistant","content":ex_ans})
    msgs.append({"role":"user","content":f"【条文】\n{t1}\n\n【陳述文】\n{t2}"})
    return msgs

def extract_label(text):
    m=re.search(r'判定[：:]\s*([YN])',text)
    if m: return m.group(1)
    matches=re.findall(r'\b([YN])\b',text)
    return matches[-1] if matches else 'N'

data=[json.loads(l) for l in open(DATA_PATH,encoding='utf-8')]

done_ids=set()
if os.path.exists(OUT_PATH):
    for l in open(OUT_PATH,encoding='utf-8'):
        done_ids.add(json.loads(l)['id'])
    print(f"Resuming — {len(done_ids)} already done")

print(f"Running STRUCTURED prompt on {len(data)} examples...")
correct=total=0
out_f=open(OUT_PATH,'a',encoding='utf-8')

for i,sample in enumerate(data):
    if sample.get('id') in done_ids: continue
    t2=sample['t2']
    retrieved,score_map=retrieve(t2,k=10)
    top_arts=sorted(retrieved&set(score_map.keys()),key=lambda x:score_map.get(x,0),reverse=True)[:5]
    extra=[a for a in retrieved if a not in top_arts][:2]
    t1='\n'.join(articles[a] for a in top_arts+extra if a in articles)

    try:
        resp=client.chat.completions.create(
            model=MODEL,
            messages=make_messages(t1,t2,structured=True),
            max_tokens=400,
            temperature=0.0,
        )
        response_text=resp.choices[0].message.content
        pred=extract_label(response_text)
    except Exception as e:
        print(f"  error at {i}: {e}"); time.sleep(30); continue

    pred_int=1 if pred=='Y' else 0
    true_label=sample.get('label')
    result={'id':sample.get('id'),'year':sample.get('year'),
            't2':t2,'pred_label':pred,'pred_int':pred_int,
            'retrieved_arts':list(retrieved),'response':response_text}
    if true_label is not None:
        result['true_label']=true_label
        result['correct']=(pred_int==true_label)
        correct+=(pred_int==true_label); total+=1

    out_f.write(json.dumps(result,ensure_ascii=False)+'\n'); out_f.flush()
    if (i+1)%20==0:
        acc_str=f" | Acc: {correct/total:.3f}" if total>0 else ""
        print(f"[{i+1}/{len(data)}]{acc_str}",flush=True)
    time.sleep(1.0)

out_f.close()
results=[json.loads(l) for l in open(OUT_PATH,encoding='utf-8')]
trues=[r['true_label'] for r in results if 'true_label' in r]
preds=[r['pred_int'] for r in results if 'true_label' in r]
if trues:
    from sklearn.metrics import accuracy_score,classification_report
    print(f"\nSTRUCTURED PROMPT FINAL ACCURACY: {accuracy_score(trues,preds):.4f}")
    print(classification_report(trues,preds,target_names=['N','Y'],digits=4))
    
    # Year breakdown
    from collections import defaultdict
    year_stats=defaultdict(lambda:{'c':0,'t':0})
    for r in results:
        if 'true_label' not in r: continue
        year_stats[r['year']]['t']+=1
        if r['correct']: year_stats[r['year']]['c']+=1
    print("\nBy year:")
    for yr in sorted(year_stats):
        s=year_stats[yr]
        print(f"  {yr}: {s['c']}/{s['t']} = {s['c']/s['t']:.3f}")
