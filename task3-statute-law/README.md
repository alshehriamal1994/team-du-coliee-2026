# Task 3, Statute Retrieval and Entailment

The task works over the Japanese Civil Code. Given a legal bar examination question, the system retrieves the relevant articles and then decides whether they entail the statement in the question. Performance is measured by accuracy over the questions.

## Result

Our best official run, DU2, answered 65 of 82 questions correctly, an accuracy of 79.3%, and placed 14th of 22 runs. After the competition, replacing the entailment model answered 75 of 82, an accuracy of 91.5%, with no change to retrieval. The post-competition figure is reported for interest and is not part of the ranked result.

## Method

Retrieval and entailment are kept separate, and most of the design effort went into keeping retrieval clean so that the entailment model could be judged on its own terms.

1. **Retrieval.** BM25 over character bigrams, which suits Japanese because it has no spaces between words, returns the top articles. A regular-expression pass then follows cross-references between articles, giving between 11 and 17 articles per question. Retrieval recall is between 0.94 and 0.98.
2. **Entailment, official.** Qwen2.5-72B reads the question and the retrieved articles and returns a structured judgement at temperature zero.
3. **Entailment, post-competition.** The same step with Qwen3-235B and an IRAC prompt, which asks the model to reason in the order Issue, Rule, Application, Conclusion. Retrieval is unchanged.

## What the results show

Because retrieval recall is already high, the entailment model is the limiting factor, and a stronger model gave a large gain. The more interesting finding is the other direction. Adding complexity hurt. Our more elaborate official run lost 3.7 points against the simpler DU2, and the nine-expert ensemble that won Task 4 reaches only 86.6% here, because the extra retrieved articles act as distractors that mislead the weaker members. On this task, restraint paid.

## Reproducing the result

Place the Task 3 data under `../data/task3/`. The statute data is in Japanese, and the translation step is documented with the scripts in `src/`. Run retrieval first, then either entailment model.
