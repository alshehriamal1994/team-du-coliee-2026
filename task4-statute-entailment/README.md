# Task 4, Statute Entailment

Given a legal statement and the relevant articles of the Japanese Civil Code, the task is to decide whether the articles entail the statement. The answer is yes or no, and performance is measured by accuracy. This is the task we won.

## Result

Our two strongest runs each answered 79 of 82 questions correctly, an accuracy of 96.3%, and took the top two places among 33 runs from 11 teams. A third run was also submitted. The best other team reached 95.1%.

## Method

The system is an ensemble of nine open-weight experts drawn from three model families. Each expert reads the statement and the articles and returns an independent yes or no. The families are diverse on purpose, because the gain comes from disagreement between them rather than from any single strong model.

The nine experts are built from these models.

- DeepSeek R1 (671B)
- Llama 4 Maverick (400B), Llama 4 Scout (109B), Llama 3.3 (70B)
- Qwen 2.5 (72B), Qwen3 (235B)

Each expert is a pairing of one of these models with a prompt strategy, and the prompts vary as deliberately as the models do. They range from a plain instruction to chain-of-thought, the IRAC legal framework (Issue, Rule, Application, Conclusion), a word-by-word check, and a self-consistency vote. The same model gives different answers depending on how it is asked, and that variation is part of what the ensemble exploits. The strategies are written out in [`prompts.md`](prompts.md).

We submitted three runs that aggregate the experts in increasing order of sophistication.

- **DU3.** A plain majority vote over the nine experts.
- **DU2.** A majority vote with a deliberation step. When the vote is close, three judge models re-read the question and revise the decision only if they agree.
- **DU1.** A hierarchical meta-ensemble that combines three sub-panels of experts.

## Why diversity wins

The case for a cross-architecture ensemble is made on the validation set. The best ensemble of a single model with several prompts of itself reaches 87.2%. A nine-expert vote across the three families reaches 91.9%, and the hierarchical aggregation reaches 93.0%. The reason is independence of error. Expert pairs from different families disagree on 17.5% of questions, against 11.4% for pairs from the same family, and a vote only helps when its members fail in different places. A single model, however large, tends to fail in the same places each time.

The analysis behind these numbers, including the single-model study and the decomposition of the ensemble's uncertainty, is in [`../docs/architecture-gap.md`](../docs/architecture-gap.md).

## Compliance

Every model is open-weight, as the rules require, and each was released before 15 July 2025, the cut-off the rules set for Tasks 3 and 4.

## Running it

The code that produces the experts and the vote is in [`src/`](src):

- `run_entailment_task4.py` runs one model as an expert and writes a Y or N for every question.
- `run_prompt_ensemble_task4.py` runs one model under several prompt strategies and votes across them.
- `run_fewshot_voting_task4.py` runs the self-consistency expert, sampling several times and taking the internal majority.
- `ensemble_predictions.py` takes the predictions of the nine experts and combines them by majority vote.

What you need first. The COLIEE data placed under `../data/task4/`, the Civil Code articles as XML, and the open-weight models downloaded locally. The larger models are demanding, so the scripts accept `--load-in-4bit` and `--load-in-8bit` for quantised loading, and a GPU is assumed.

A single expert:

```
python src/run_entailment_task4.py \
  --model-path /path/to/a/local/model \
  --civil-xml ../data/task4/civil_code.xml \
  --input-jsonl ../data/task4/test.jsonl \
  --run-tag EXPERT1 \
  --output runs/expert1.txt \
  --use-chat-template --load-in-4bit
```

The nine experts combined into the majority-vote run, which is DU3:

```
python src/ensemble_predictions.py \
  --inputs runs/expert1.txt runs/expert2.txt runs/expert3.txt \
           runs/expert4.txt runs/expert5.txt runs/expert6.txt \
           runs/expert7.txt runs/expert8.txt runs/expert9.txt \
  --output runs/DU3.txt --run-tag DU3 --tie-break Y
```

The two stronger runs build on this. DU2 adds the deliberation step, in which three judge models re-read the questions where the vote is close, and DU1 aggregates the experts hierarchically across three sub-panels. The deliberation prompt and the composition of the sub-panels are described in [`prompts.md`](prompts.md) and in the system description in our submission.

DU3, the nine-expert majority vote, is fully reproducible from this directory. DU1 and DU2 are not. Their deliberation and judge step and their hierarchical sub-panel aggregation are described here, but the orchestration code is not shipped. Reproducing the exact figures requires the same nine experts, that aggregation, and substantial GPU memory for the largest models.

Dependencies are in [`requirements.txt`](requirements.txt).
