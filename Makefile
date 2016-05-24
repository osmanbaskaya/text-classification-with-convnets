SHELL := /bin/bash

NUM_FOLD = 10
PROBLEM_TYPE=regression
USE_PRETRAINED_EMBEDDINGS=True
MODEL_DIR=""

cross-validate-%: datasets/%
	python evaluate.py $< ${NUM_FOLD} ${PROBLEM_TYPE} ${USE_PRETRAINED_EMBEDDINGS} | tee $@.score

pretrained-models:
	mkdir $@

pretrained-models/%: datasets/% pretrained-models
	mkdir $@
	python build_model.py $< $@ ${PROBLEM_TYPE} ${USE_PRETRAINED_EMBEDDINGS}


%.pred: datasets/% ${MODEL_DIR}
	python predict.py $^ > $@
	wc $@

.SECONDARY:
