#   Text Classification with ConvNets (Basic Keras Practices)
CNN classifier for Formality in text.


### Cross Validation

In order to run cross validation on a dataset, you can just run:

```bash
make cross-validate-formality.lahiri.dataset NUM_FOLD=10 PROBLEM_TYPE=regression
```

`PROBLEM_TYPE` defines your problem, either regression or classification. Labels for `formality.lahiri.dataset` between -3 to 3 and real-valued. So, you should give `PROBLEM_TYPE=regression`.
`NUM_FOLD` determines how many fold you run for evaluation of the dataset. Important to note that dataset should be in datasets/ directory.

Example for a classification problem:

```
make cross-validate-formality.lahiri.classes.clean.dataset NUM_FOLD=10 PROBLEM_TYPE=classification
```

### Build and Save Model

Assume that we have formality datasets for emails and it is located in `datasets/formality-email`. You can run the following command; build and save a model in `pretrained-models` directory.

```bash
make pretrained-models/formality-email USE_PRETRAINED_EMBEDDINGS=False
```

If you want to use pretrained word embeddings (e.g., Word2Vec) then you can run this:

```bash
make pretrained-models/formality-email
```

### Prediction with Pretrained Models

Assume that you trained a regressor (or classifier) by using formality email datasets as above. By using this pretrained model, we can predict formality scores of the sentences in news domain. In the `datasets` directory, there is a dataset named `formality-news`. Let's use it as our test data.

```bash
make formality-answers.pred MODEL_DIR=pretrained-models/formality-email USE_PRETRAINED_EMBEDDINGS=False
```

If you want to use Word2Vec and if you haven't built any model yet, you may remove providing `USE_PRETRAINED_EMBEDDINGS=False` (since default is `True`):

```
make formality-answers.pred MODEL_DIR=pretrained-models/formality-email
```

Even you haven't run the command for build a model, `Makefile` finds the dependency path and run the necessary commands for you. Also, it is important that training data for the model building should be somewhat similar with the test data. News and email domains can be significantly different with each other. I used these two datasets just to illustrate the commands.
