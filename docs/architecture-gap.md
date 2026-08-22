# The architecture gap

This note explains, with the validation numbers, why the Task 4 result comes from the diversity of the experts rather than from the size of any one model. The headline figures are repeated in the paper, and the fuller analysis, including the single-model study and the decomposition of the ensemble's uncertainty, is in the paper itself.

## The measurement

All three figures are accuracies on the validation set, the average of the H30, R01, and R02 splits.

| System | Validation accuracy |
| --- | --- |
| Best ensemble of a single model with several prompts of itself | 87.2% |
| Nine-expert vote across three model families | 91.9% |
| Hierarchical meta-ensemble of the nine experts | 93.0% |

Ensembling one model with several prompts of itself reaches 87.2% and then stops improving. Putting nine experts from three different families to the vote adds 4.7 points. Aggregating those experts hierarchically adds a further 1.1.

## Why it happens

A vote only helps when its members fail in different places. If every expert makes the same mistakes, no amount of voting recovers them. The experts in this system are chosen to be diverse on purpose, in both the model family and the prompt strategy, so that their errors are as independent as possible.

That independence is measurable. Expert pairs drawn from different model families disagree on 17.5% of questions, against 11.4% for pairs drawn from the same family. The cross-family pairs disagree more, and it is precisely that disagreement that the vote turns into a gain. A single model, however large, tends to be wrong in the same places each time, which is why scaling one model up does not close the gap.

## What this means in practice

For a task like statute entailment, where the answer is short and well defined, the useful lever is not a bigger model but a more diverse panel of moderate ones. The prompt strategies that widen that diversity are set out in [`../task4-statute-entailment/prompts.md`](../task4-statute-entailment/prompts.md).
