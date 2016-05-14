# formality-classifier-cnn
CNN classifier for Formality in text.

In order to run cross validation on a dataset, you can just run:

```
make cross-validate-formality.lahiri.dataset NUM_FOLD=10 PROBLEM_TYPE=regression
```

`PROBLEM_TYPE` defines your problem, either regression or classification. Labels for `formality.lahiri.dataset` between -3 to 3 and real-valued. So, you should give `PROBLEM_TYPE=regression`.
`NUM_FOLD` determines how many fold you run for evaluation of the dataset. Important to note that dataset should be in datasets/ directory.

Example for a classification problem:

```
make cross-validate-formality.lahiri.classes.clean.dataset NUM_FOLD=10 PROBLEM_TYPE=classification
```
