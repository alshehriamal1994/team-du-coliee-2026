# Results in full

All figures are taken from the official COLIEE 2026 evaluation. The official results are the only ranked figures; every post-competition figure was obtained after the deadline with no change to the system architecture, is marked as such, and is reported for interest only.

## Task 1, legal case retrieval

| Run | F1 | Precision | Recall | Note |
| --- | --- | --- | --- | --- |
| DU3 | 0.314 | 0.295 | 0.337 | official, rank 11 of 54 runs |
| DU9 | 0.346 | 0.324 | 0.370 | post-competition, deeper reranker |

The query documents have a median length of 4,573 tokens, which is well beyond the 512-token window of the encoders we tried. Cross-encoders and zero-shot language models truncated the documents and did poorly. A learning-to-rank model over hand-built features did better, and that is the system we submitted.

## Task 2, legal case entailment

| Run | F1 | Note |
| --- | --- | --- |
| DU3 | 0.343 | official, rank 22 of 35 runs |
| post-competition | 0.555 | one change to the prompt, same models and pipeline |

The task asks which paragraphs of a candidate case entail a held-out fragment of the decision. Our official prompt asked the model to select at most one paragraph. Cases in the test set contain on average 2.94 entailing paragraphs, so a single-selection rule caps F1 at 0.508 by construction. Changing the instruction to select every entailing paragraph raised F1 to 0.555, above the best official entry of 0.490. The lesson is recorded honestly here because the improvement is a prompt correction made after the fact, not part of the ranked result.

## Task 3, statute retrieval and entailment

| System | Correct | Accuracy | Note |
| --- | --- | --- | --- |
| DU2 | 65 of 82 | 79.3% | official, rank 14 of 22 runs |
| Qwen3-235B with IRAC | 75 of 82 | 91.5% | post-competition, model swap only |
| Nine-expert ensemble | 71 of 82 | 86.6% | the Task 4 system applied here |

Retrieval recall on this task is between 0.94 and 0.98, so the entailment model, not retrieval, is the limiting factor. Two findings are worth stating plainly. First, a stronger entailment model with the same clean retrieval gave a large gain. Second, added pipeline complexity hurt. Our more elaborate run lost 3.7 points against the simpler DU2, and the nine-expert ensemble that won Task 4 underperforms a single Qwen3-235B here, because distractor articles mislead the weaker members.

## Task 4, statute entailment

| Run | Validation | Test | Note |
| --- | --- | --- | --- |
| DU3, nine-expert majority vote | 91.9% | 93.9% | |
| DU2, with deliberation | 92.6% | 96.3% | first place |
| DU1, hierarchical meta-ensemble | 93.0% | 96.3% | first place |

Both DU1 and DU2 reached 96.3% (79 of 82) on the unseen test set and took the top two places among 33 runs from 11 teams.

The gain comes from disagreement between architectures, not from scale. On the validation set the best ensemble of a single model with itself reaches 87.2%. A nine-expert vote across three model families reaches 91.9%, and a hierarchical aggregation of those experts reaches 93.0%. Expert pairs from different families disagree on 17.5% of questions, against 11.4% for pairs from the same family, and that independence is the precondition that lets voting help.

All nine experts are open-weight models, as the rules require, and for Tasks 3 and 4 every model was released before 15 July 2025.

## Pilot task, legal judgment prediction (tort)

| Configuration | Tort accuracy | Rationale F1 | Note |
| --- | --- | --- | --- |
| Full five-view ensemble with bridge | 73.1% | 68.2% | our system |
| Single BERT only | 70.9% | 64.9% | ablation |
| Five-view, no verdict bridge | 70.5% | 66.9% | ablation |

The pilot entry was submitted under the wrong run mode and the organisers listed it as unofficial. We report it for completeness. Its tort accuracy is above every official entry, and its rationale F1 matches the best official figure. The claim-to-verdict bridge recovers information lost when long cases are truncated at 512 tokens, which affects 41% of the cases, and it accounts for 2.8 points of the tort accuracy on its own.
