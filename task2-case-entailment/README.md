# Task 2, Legal Case Entailment

Given a query case and a held-out fragment of its decision, the task is to identify which paragraphs of a candidate case entail that fragment. Performance is measured by F1 over the selected paragraphs.

## Result

Our official run reached an F1 of 0.343 and placed 22nd of 35 runs. After the competition, a single change to the prompt raised F1 to 0.555, above the best official entry of 0.490. The improvement used the same models and the same pipeline. It is reported here as a finding, not as a ranked result.

## Method

The pipeline retrieves and reranks candidate paragraphs and then asks an ensemble of language models which of them entail the fragment.

1. BM25 returns the top 100 candidate paragraphs.
2. A MonoT5 reranker and a Qwen3 reranker reduce these to the top 20.
3. Three open-weight language models judge entailment with a few-shot prompt.
4. A majority vote across the three models gives the final selection.

## The prompt study

The official prompt asked the model to select at most one paragraph, or none. The test cases contain on average 2.94 entailing paragraphs, so a single-selection rule cannot exceed an F1 of 0.508 however good the model is. Of the 230 gold paragraphs our official run missed, 122 were lost to that constraint alone rather than to any error of judgement.

Changing the instruction to select every paragraph that entails the fragment removed the cap and raised F1 to 0.555. The pipeline and the models were untouched. The point is narrow but worth making. On this task the wording of the instruction mattered more than the choice of model.

## Reproducing the result

Place the Task 2 data under `../data/task2/`, then run the pipeline. Both prompt variants are kept side by side in `src/`, since the contrast between them is the result.
