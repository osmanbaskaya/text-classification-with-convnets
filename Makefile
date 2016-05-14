NUM_FOLD = 10
PROBLEM_TYPE=regression

cross-validate-%: datasets/%
	python evaluate.py $< ${NUM_FOLD} ${PROBLEM_TYPE} | tee $@.score
