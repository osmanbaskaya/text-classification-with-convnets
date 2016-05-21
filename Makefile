NUM_FOLD = 10
PROBLEM_TYPE=regression
USE_PRETRAINED_EMBEDDINGS=True

cross-validate-%: datasets/%
	python evaluate.py $< ${NUM_FOLD} ${PROBLEM_TYPE} ${USE_PRETRAINED_EMBEDDINGS} | tee $@.score
