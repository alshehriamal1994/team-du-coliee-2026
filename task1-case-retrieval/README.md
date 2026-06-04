# Task 1, Legal Case Retrieval

Given a query case from Canadian case law, the task is to retrieve the earlier cases it relies on from a large candidate pool. Performance is measured by F1 over the set of cases returned.

## Result

Our submitted run, DU3, reached an F1 of 0.314 and placed 11th of 54 runs. A deeper version of the reranker built after the competition, DU9, reached 0.346. The post-competition figure is not part of the ranked result and is reported only for interest.

## Method

The query documents are long, with a median of 4,573 tokens. That single fact shaped the design. It is well past the 512-token window of the sentence encoders and cross-encoders we tried, so those models truncated the cases and lost the parts that mattered. We therefore treat retrieval as a ranking problem over interpretable features rather than as a neural matching problem.

The pipeline has three stages.

1. **Candidate pool.** BM25 and a dense retriever are run separately and their top results are merged. On the development set the merged pool contains the correct case 99.1% of the time, which sets the ceiling for the reranker.
2. **Reranking.** A LightGBM model is trained with the LambdaRank objective over thirty-four features. They fall into four groups: eleven retrieval scores, sixteen structural features of the documents, three measures of citation authority, and four temporal features. The strongest single feature is the reciprocal-rank fusion of the two retrievers.
3. **Post-filter.** Self-matches and candidates dated after the query are removed. On the development set this adds 2.7 points of F1.

The contribution of each group is visible in a development ablation, which moves from 16.5 with BM25 alone to 35.7 with all features and the full training data.

## Running it

The learning-to-rank code is in [`src/`](src). It works on candidate features produced by the retrieval stage:

- `train_du4.py` and `train_du7.py` train the LightGBM ranker under the configurations we used.
- `train_citation_chains.py` adds the citation-chain features.
- `tune_step8_and_seeds.py` tunes the fusion step and the seed ensemble.
- `vote_ensemble.py` combines several trained runs.
- `evaluate.py` reports micro precision, recall, and F1 against the gold labels.

**Expected inputs.** The learning-to-rank scripts read pre-built candidate feature caches: the `.npz` files holding the arrays `X`, `labels`, `qids`, `cids`, and `feature_names`. The post-filter step (step8) takes the top-50 candidates and applies self-match and future-date filtering. The `FINAL_SUBMISSION/` directory holds the prediction files and gold labels, and the scripts write their outputs under `runs/`. All of these are derived from the licensed COLIEE data and are not shipped with this repository. The retrieval stage and the thirty-four features that build the candidates are the ones described above.

Dependencies are in [`requirements.txt`](requirements.txt).

## A note on what did not work

Neural rerankers are the obvious choice for retrieval and they were not competitive here. The length of the documents is the reason. This is the clearest case in our five tasks of a problem where hand-built features earn their place.
