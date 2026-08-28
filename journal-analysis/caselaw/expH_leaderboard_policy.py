"""Experiment H - official-leaderboard policy decomposition (both tasks).

For every official run, average returned-set size is derivable from public P/R alone:
tp = R * total_gold; n_pred = tp / P; size = n_pred / n_queries. We test how strongly F1
tracks answer-set policy - measured as distance of a run's size from the gold mean -
across the whole field. Tables fetched from coliee.org/COLIEE2026/results/task{1,2} on
2026-07-22 (verified against our archived per-run numbers). Zero-score runs excluded
(pipeline failures, not policy). Writes expH_numbers.json + expH_runs.csv. 2026-07-22.
"""
import csv, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent

T1 = [  # rank, run, team, P, R, F1  (54 runs; gold: 1750 cites / 400 queries, mean 4.375)
 (1,"submission_2","NOWJ",.4235,.4206,.4220),(2,"submission_3","NOWJ",.4140,.4263,.4200),
 (3,"random_forest","JNLP",.4341,.3931,.4126),(4,"ensemble","JNLP",.4174,.3914,.4040),
 (5,"xgboost","JNLP",.4059,.3943,.4000),(6,"submission_1","NOWJ",.3772,.3994,.3880),
 (7,"submission_sil2","SIL",.4186,.3600,.3871),(8,"submission_sil1","SIL",.3856,.3766,.3810),
 (9,"task_1_mezzanino","mezza",.3161,.3429,.3289),(10,"3.intit_bm25_year","INTIT",.3249,.3086,.3165),
 (11,"du3","DU",.2945,.3366,.3141),(12,"du2","DU",.2940,.3360,.3136),
 (13,"1.intit_bm25_pys","INTIT",.3309,.2851,.3063),(14,"du1","DU",.2845,.3251,.3035),
 (15,"task1_flnlpltr","FLNLP",.2619,.3337,.2935),(16,"2.intit_bm25_judge","INTIT",.3147,.2611,.2854),
 (17,"task1_flnlpembed","FLNLP",.2540,.2903,.2709),(18,"ua_run3","UA",.2038,.3851,.2666),
 (19,"ua_run2","UA",.2056,.3714,.2647),(20,"ucdcs3","UCD-CS",.2480,.2834,.2645),
 (21,"ucdcs1","UCD-CS",.2131,.3354,.2607),(22,"ua_run1","UA",.2114,.3297,.2576),
 (23,"ucdcs2","UCD-CS",.2405,.2749,.2565),(24,"task1_run2_results","JUNLLP",.2193,.2977,.2525),
 (25,"task1_run1_results","JUNLLP",.2369,.2680,.2515),(26,"task1_flnlpbm25","FLNLP",.2340,.2674,.2496),
 (27,"task1_run3_results","JUNLLP",.2002,.3171,.2455),(28,"task-1-ualbany","74688",.2130,.2611,.2346),
 (29,"run1","UB2026",.2338,.2137,.2233),(30,"sach_task1_run2","Sach",.1631,.3531,.2231),
 (31,"sach_task1_run1","Sach",.1683,.3177,.2200),(32,"bjpwh3","BJPWH",.1510,.3451,.2101),
 (33,"bjpwh2","BJPWH",.1500,.3429,.2087),(34,"run2","UB2026",.2006,.1834,.1916),
 (35,"run2recall","ABAI",.1596,.1989,.1771),(36,"run1balanced","ABAI",.1945,.1463,.1670),
 (37,"task1_aiirqwen","AIIRLab",.2642,.1063,.1516),(38,"bosch_task1_run","bosch",.0892,.4103,.1466),
 (39,"task1_submission_v9","KeioAndrewShin",.1527,.1263,.1383),
 (40,"task1_submission_v8","KeioAndrewShin",.1150,.1531,.1314),
 (41,"run3precis","ABAI",.2200,.0931,.1309),
 (42,"task1_submission_v7","KeioAndrewShin",.1544,.1046,.1247),
 (43,"bjpwh1","BJPWH",.0870,.1937,.1201),(44,"task1_aiirparanemo","AIIRLab",.1055,.1206,.1125),
]
T2 = [  # rank, run, team, P, R, F1  (35 runs; gold: 294 paras / 100 cases, mean 2.94)
 (1,"task2_run2","IAI",.4501,.5374,.4899),(2,"task2_aiirfusion3","AIIRLab",.5120,.4354,.4706),
 (3,"submission (3)","JNLP",.4174,.4898,.4507),(4,"task2_run3","IAI",.4769,.4218,.4477),
 (5,"submission (2)","JNLP",.3879,.5238,.4457),(6,"nowj003","NOWJ",.7037,.3231,.4429),
 (7,"task2_run1","IAI",.4877,.4048,.4424),(8,"task2_aiirfusion2","AIIRLab",.4257,.4286,.4271),
 (9,"result_run1_ua","UA",.4711,.3878,.4254),(10,"submission (1)","JNLP",.3381,.5680,.4239),
 (11,"nowj002","NOWJ",.7034,.2823,.4029),(12,"submission (1)","JUNLLP",.6721,.2789,.3942),
 (13,"submission (2)","JUNLLP",.6721,.2789,.3942),
 (14,"bosch_task_2_submission (1)","bosch",.3900,.3980,.3939),
 (15,"task2_clsop3d","ClaUSurf",.6357,.2789,.3877),
 (16,"bosch_task_2_submission (2)","bosch",.2634,.6871,.3808),
 (17,"task2_cls4b27bp11","ClaUSurf",.7400,.2517,.3756),(18,"nowj001","NOWJ",.7604,.2483,.3744),
 (19,"task2_clsp2d","ClaUSurf",.6525,.2619,.3738),(20,"result_run2_ua","UA",.5882,.2721,.3721),
 (21,"result_run3_ua","UA",.3592,.3469,.3529),(22,"task2_du3","DU",.6907,.2279,.3427),
 (23,"task2_du2","DU",.7529,.2177,.3377),(24,"task2_aiirdeberta","AIIRLab",.5750,.2347,.3333),
 (25,"ualbanyllm","74688",.6500,.2211,.3299),(26,"task2_du1","DU",.6632,.2143,.3239),
 (27,"task2_submission_uottawanii2","KeioAndrewShin",.4795,.2381,.3182),
 (28,"task2_submission_uottawanii3","KeioAndrewShin",.4600,.2347,.3108),
 (29,"task2_submission","CityUMO",.6000,.2041,.3046),(30,"ualbanybm25","74688",.5200,.1769,.2640),
 (31,"task2_submission_uottawanii1","KeioAndrewShin",.4275,.1905,.2635),
 (32,"sach_task2_run1","Sach",.3929,.1497,.2167),
]

def spearman(a, b):
    # Ties must take the average rank. argsort-of-argsort assigns tied values
    # arbitrary distinct ranks, and the leaderboards are full of ties (eight
    # Task 1 runs return exactly five answers per query, four Task 2 runs one).
    from scipy.stats import rankdata
    ra = rankdata(a).astype(float)
    rb = rankdata(b).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])

def analyse(rows, total_gold, n_q, gold_mean, name):
    out = []
    for rank, run, team, P, R, F1 in rows:
        tp = R * total_gold
        size = tp / P / n_q
        out.append({'rank': rank, 'run': run, 'team': team, 'P': P, 'R': R, 'F1': F1,
                    'size': round(size, 2),
                    'log_dist': round(abs(math.log(size / gold_mean)), 3)})
    f1 = np.array([r['F1'] for r in out])
    size = np.array([r['size'] for r in out])
    dist = np.array([r['log_dist'] for r in out])
    top = [r['size'] for r in out[:5]]
    rest = [r['size'] for r in out[5:]]
    return out, {
        'n_runs_scored': len(out), 'gold_mean': gold_mean,
        'spearman_F1_vs_size': round(spearman(f1, size), 3),
        'spearman_F1_vs_logdist_from_gold': round(spearman(f1, -dist), 3),
        'top5_size_mean': round(float(np.mean(top)), 2),
        'rest_size_mean': round(float(np.mean(rest)), 2),
        'note': f'{name}: F1 vs -|log(size/gold_mean)| - positive means closer to gold size, higher F1'}

t1_rows, t1_stats = analyse(T1, 1750, 400, 4.375, 'T1')
t2_rows, t2_stats = analyse(T2, 294, 100, 2.94, 'T2')

with open(HERE / 'expH_runs.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['task', 'rank', 'run', 'team', 'P', 'R', 'F1', 'avg_set_size', 'log_dist_from_gold'])
    for r in t1_rows:
        w.writerow(['T1', r['rank'], r['run'], r['team'], r['P'], r['R'], r['F1'], r['size'], r['log_dist']])
    for r in t2_rows:
        w.writerow(['T2', r['rank'], r['run'], r['team'], r['P'], r['R'], r['F1'], r['size'], r['log_dist']])

res = {'meta': {'date': '2026-07-22', 'source': 'coliee.org/COLIEE2026/results/task{1,2}, fetched 2026-07-22',
                'excluded': 'zero-score runs (10 in T1, 3 in T2) = pipeline failures, not policy',
                'du_sizes': {'T1_du3': next(r['size'] for r in t1_rows if r['run'] == 'du3'),
                             'T2_du3': next(r['size'] for r in t2_rows if r['run'] == 'task2_du3')}},
       'task1': t1_stats, 'task2': t2_stats}
(HERE / 'expH_numbers.json').write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
