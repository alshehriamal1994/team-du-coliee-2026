# Team DU at COLIEE 2026

[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-68246D?style=flat-square)](LICENSE)
[![COLIEE 2026](https://img.shields.io/badge/COLIEE-2026-68246D?style=flat-square)](https://coliee.org/COLIEE2026/overview)
![Task 4: 1st of 33](https://img.shields.io/badge/Task_4-1st_of_33-C8A415?style=flat-square)
![Python 3.11](https://img.shields.io/badge/Python-3.11-68246D?style=flat-square)

**First place in Task 4, statute entailment, with 96.3% accuracy.** This repository contains Durham University's code for all five tasks of the 2026 Competition on Legal Information Extraction and Entailment (COLIEE), held alongside the [21st International Conference on Artificial Intelligence and Law (ICAIL 2026)](https://site.smu.edu.sg/icail-2026) in Singapore. The work is described in our paper, *Cross-Architecture LLM Ensembles, Feature-Based Reranking and Retrieval-Augmented Prompting for Legal Information Processing*.

The single idea running through the five tasks is that no one method wins everywhere. Different legal tasks reward different inductive biases. A cross-architecture ensemble of open-weight language models won statute entailment outright. Long-document retrieval was carried by hand-built features rather than neural rerankers. On statute law a stronger model with clean retrieval beat a more elaborate pipeline. The competition rules permit only models whose weights are publicly available and forbid closed models such as GPT-4o and Gemini, so every system here uses open-weight models. We used no closed model at any point.

## The competition

COLIEE is the [Competition on Legal Information Extraction and Entailment](https://coliee.org/COLIEE2026/overview), an annual benchmark of legal case retrieval, statute retrieval, and legal entailment. The 2026 edition is held alongside the [21st International Conference on Artificial Intelligence and Law (ICAIL 2026)](https://site.smu.edu.sg/icail-2026), at the Yong Pung How School of Law, Singapore Management University, from 8 to 12 June 2026. ICAIL is organised under the International Association for Artificial Intelligence and Law, and this is the first time it has been held in Asia. COLIEE 2026 ran four shared tasks on case law and statute law, with a pilot task on tort prediction. We entered all of them.

## Results

| Task | What it is | Result | Official rank | Post-competition |
| --- | --- | --- | --- | --- |
| 1 | Legal case retrieval | F1 0.314 | 11 of 54 runs | F1 0.346 |
| 2 | Legal case entailment | F1 0.343 | 22 of 35 runs | F1 0.555 |
| 3 | Statute retrieval and entailment | 79.3% | 14 of 22 runs | 91.5% |
| 4 | Statute entailment | 96.3% accuracy | 1st of 33 runs | n/a |
| Pilot | Legal judgment prediction (tort) | 73.1% tort accuracy, 68.2% rationale F1 | unofficial (run-mode error) | n/a |

Task 4 is our principal result. Our two strongest runs each reached 96.3% (79 of 82) on the unseen test set and took the top two places among 33 runs from 11 teams; the best competing team reached 95.1%. A third run was also submitted. The pilot entry was submitted in the wrong run mode and the organisers listed it as unofficial; although unranked, its tort accuracy exceeds every official entry and its rationale F1 equals the best official figure.

The post-competition column reports improvements made after the deadline with no change to the system architecture. They are not ranked figures, they are labelled as such throughout, and the full table with all runs is in [`docs/results.md`](docs/results.md).

## What we found

On Task 2, changing the instruction from "select at most one paragraph" to "select every paragraph that entails the fragment" raised F1 by 0.212, past the best official entry, with nothing else altered. On this task the wording of the instruction proved more consequential than the choice of model.

On statute law, more pipeline made things worse. With retrieval recall already above 0.94, a stronger entailment model beat our more elaborate run by a wide margin, and the elaborate run itself lost 3.7 points to a simpler one.

The system that won one task lost another. The nine-expert ensemble that took first place on Task 4 is beaten by a single Qwen3-235B on Task 3, because the extra retrieved articles act as distractors that mislead the weaker members.

On Task 4 the gain came from disagreement between model families rather than from any single large model.

## The winning system

<p align="center"><img src="assets/figures/task4-ensemble.gif" alt="The Task 4 ensemble in motion, from the statement and articles through nine experts casting independent yes or no votes to the majority vote, the meta-ensemble, and the result" width="880"></p>

Nine open-weight experts from three families each return an independent yes or no, and a hierarchical meta-ensemble combines them. The experts differ in prompt as well as in model, from a plain instruction to chain-of-thought, the IRAC legal framework, and a word-by-word check, and that diversity of prompting is set out in [`task4-statute-entailment/prompts.md`](task4-statute-entailment/prompts.md). The case for diversity is clearest on the validation set, where a single model ensembled with itself plateaus well below the cross-architecture vote.

<p align="center"><img src="assets/figures/architecture-gap.gif" alt="Validation accuracy rising from 87.2 for a single-model ensemble to 91.9 for the nine-expert vote and 93.0 for the meta-ensemble" width="700"></p>

## The pilot pipeline

The pilot tort task is unofficial, submitted under the wrong run mode, but its tort accuracy is above every official entry and its rationale F1 matches the best. Five views read each case, a stacker weights them, a claim-to-verdict bridge repairs what truncation loses on the long cases, and a coherence check removes contradictions before the decision is fixed. The detail is in [`pilot-tort-prediction/`](pilot-tort-prediction/).

<p align="center"><img src="assets/figures/pilot-pipeline.gif" alt="The pilot pipeline in motion: five views read each tort case, a stacker weights them, a claim-to-verdict bridge repairs the decision, a coherence check removes contradictions, and the result reaches 73.1 per cent, above every official entry" width="880"></p>

## Journal-extension analyses

The `journal-analysis/` folder contains the post-hoc analysis code and result
files behind our journal extension of the proceedings paper (Review of
Socionetwork Strategies, COLIEE 2026 special issue), together with the
per-expert prediction outputs the analyses run on. See the README in that
folder for what is included and how to rerun it.

## Using this repository

This repository is the reference record of the methods behind our paper. It is meant to be read alongside the paper and the per-task READMEs, and drawn on for the system designs, the exact prompts, the feature definitions and the evaluation procedures. It is not a turnkey package that can be cloned and run end to end, and it is worth being candid about why.

There are three reasons a reader cannot simply clone and reproduce our figures.

First, the data. The COLIEE 2026 datasets are released under an agreement with the organisers and may not be redistributed, so nothing under `data/` is included here. A reader who has registered with COLIEE and obtained the data can place it as described in [`data/README.md`](data/README.md), after which the data-dependent scripts will run.

Second, the models and the hardware. The statute and case systems call open-weight models, several of them very large, either downloaded locally or served through OpenRouter. Reproducing the Task 4 ensemble or the post-competition Task 3 figure needs substantial GPU memory, and where a model is served through OpenRouter the reader must supply their own key. Quantised loading is supported but does not remove the demand.

Third, the intermediate artefacts. To keep the repository to a readable size, and because they derive from the licensed data, we do not ship the pre-built feature caches, reranker score caches, trained checkpoints or prediction files that several scripts read as input. Each task README states which inputs its scripts expect and how they are produced.

What can be reproduced therefore depends on what the reader brings. With the COLIEE data and adequate GPU resources, the open-weight pipelines and the prompt studies can be re-run, and the Task 4 nine-expert majority vote (DU3) is fully reproducible from its directory. The two stronger Task 4 runs, DU1 and DU2, and the pilot solver depend on orchestration and trained artefacts that are described in full but not all shipped here. In every case the methods, the prompts and the evaluation are given in enough detail to be reimplemented independently, which is the most useful form of reproducibility for work of this kind.

## Layout

```
task1-case-retrieval/      learning-to-rank over thirty-four features
task2-case-entailment/     retrieval, reranking, few-shot ensemble, prompt study
task3-statute-law/         character-bigram retrieval and statute entailment
task4-statute-entailment/  the nine-expert cross-architecture ensemble
pilot-tort-prediction/     five-view ensemble with a claim-to-verdict bridge
docs/                      results in full and the architecture-gap analysis
data/                      how to obtain the COLIEE data (the data is not included)
```

Each task folder has its own README with the method, the commands to reproduce the result, and the numbers a correct run should produce.

## Citing this work

If you use this code, please cite the paper. The reference is in [`CITATION.cff`](CITATION.cff), and GitHub will show a "Cite this repository" button. The paper is to appear in the COLIEE 2026 proceedings, and the full bibliographic details will be added here once they are published.

## Licence

MIT. See [`LICENSE`](LICENSE).

## Authors

Amal Saad Alshehri, Nelly Bencomo, and Amir Atapour-Abarghouei. Department of Computer Science, Durham University, United Kingdom.
