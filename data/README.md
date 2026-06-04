# Data

The COLIEE datasets are not included in this repository. They are provided under an agreement with the competition organisers and may not be redistributed.

## Obtaining the data

Register through the official COLIEE site at https://coliee.org/COLIEE2026/ and follow the instructions there. Registration gives access to the training and test data for the case law tasks (1 and 2), the statute law tasks (3 and 4), and the pilot tort task.

A note on language. The statute law data for Tasks 3 and 4 is provided in Japanese only. The competition allows machine translation of the articles and queries into a working language, and our statute systems operate on translated text.

## Where to put it

Once you have the data, place each task's files under this directory in the layout each task README expects:

```
data/
  task1/
  task2/
  task3/
  task4/
  pilot/
```

The task READMEs state the exact filenames they read. Nothing under `data/` is tracked by git apart from this file.
