# Journal article analyses

The two journal articles extending our COLIEE 2026 proceedings paper each rest
on a set of post-hoc analyses of frozen prediction files. This folder holds
those analyses, one subfolder per article.

- [`statute/`](statute/) — *Model Selection and Majority Voting in
  Heterogeneous Large Language Model Ensembles for Legal Textual Entailment on
  Japanese Statute Law* (Tasks 3 and 4).
- [`caselaw/`](caselaw/) — *Learning to Rank and Learning to Stop: What Each Is
  Worth in Legal Case Retrieval and Entailment* (Tasks 1 and 2).

Every number reported in either article traces to a result file here, written
by the script beside it. Scripts that recompute from raw predictions carry
verification gates: before computing anything new, each must reproduce the
already published figures it depends on, and it stops rather than writes if any
fails to match. To check a reported number, find its result file in the
relevant subfolder, confirm the value, and rerun the script that writes it.

The competition systems themselves live in the task folders of this repository.
The COLIEE datasets are available from the organisers under the competition's
data-use agreement and are not redistributed here.
