# Prompt strategies for Task 4

The ensemble runs the same models under several different prompts. The diversity of prompting is deliberate. The same model gives different answers depending on how it is asked, and that variation is part of what makes the vote work. A model that is wrong under one framing is often right under another, and the disagreement is what a majority vote can exploit.

The prompts are written in Japanese, because the data is the Japanese Civil Code. The English translations below are ours and are faithful to the Japanese we ran. The Japanese originals are in the code. Every prompt ends by asking the model to output only Y or N on the last line, and each notes that the answers are split roughly evenly between Y and N so the model does not drift towards one label.

## 1. Standard

You are a legal expert specialising in the Japanese Civil Code. Based solely on the given article or articles, judge whether the statement logically follows.

Procedure: read the requirements and conditions of the article precisely; check exception clauses and provisos; verify negation consistency; check quantifier and scope expressions; and judge Y if the statement follows as a logical consequence of the article, even where it is not stated word for word.

## 2. Chain-of-thought

The same expert framing, but the model is asked to work through fixed steps before answering. List the requirements the article defines. List the claims the statement makes. Check each requirement against the statement one by one. Check any provisos or exceptions. Pay particular attention to limiting words such as cannot, shall not, only, and must. Then give an overall judgement.

## 3. IRAC

The IRAC method is a standard legal analysis taught in law schools. The model is asked to reason in four phases. Issue, identify what the statement claims. Rule, quote the relevant statutory provisions precisely. Application, compare the article's requirements with the statement one by one. Conclusion, decide whether the statement follows.

## 4. Meticulous

A word-by-word verification for the hardest questions. The model lists each requirement and each claim, maps one to the other, and then checks a specific set of traps with particular care: the exact correspondence of negations, whether quantifiers such as all, only, and any match the article, the presence of provisos and exceptions, whether subjects and objects are identical in the article and the statement, whether the statement assumes anything not written in the article, and technical distinctions such as possession against custody and good faith against bad faith. Finally it compares the wording character by character.

## 5. Concise

A deliberately minimal prompt. The model is given the articles and the statement and asked to judge, based on the articles alone, whether the statement is logically correct, and to output Y or N. The short prompt is included on purpose, since it sometimes disagrees with the elaborate ones and adds to the diversity of the panel.

## 6. Self-consistency

This strategy uses the Standard prompt but runs it three times at a temperature of 0.7, which introduces some randomness, and then takes the internal majority of the three answers. If two runs say Y and one says N, the expert returns Y. It captures the model's own uncertainty on borderline questions.

## Old-law-aware (R01 only)

The R01 split is drawn from a 2019 examination, and several articles were amended by the 2020 reform of the Civil Code. For that split only, a short temporal note is prepended to the Standard prompt, telling the model to judge by the pre-reform text and listing the articles whose content changed. For example, the maximum term of a lease under the old Article 604 was twenty years, which the reform raised to fifty. Without this note a current model answers by the new law and is marked wrong on an old question.

## Deliberation

The DU2 run adds a deliberation step on top of the vote. When the experts are close to evenly split, three judge models re-read the question under a separate deliberation prompt and the decision is revised only if they agree. The deliberation prompt is in the code alongside the strategies above.
