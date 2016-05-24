import os
import sys
import ujson
import codecs
from keras.models import model_from_json
from data import testset_read

test_fn = sys.argv[1]
model_dir = sys.argv[2]

model_weight_file = os.path.join(model_dir, 'weights.h5')
arch_file = os.path.join(model_dir, 'architecture.json')
word_idx_file = os.path.join(model_dir, 'word_idx.json')

# TODO: Put these in a config file.
batch_size = 32
maxlen = 25


# Read model architecture
model = model_from_json(open(arch_file).read())

# Read model weights
model.load_weights(model_weight_file)

# Read vocabulary (word_idx dict file)
word_idx = ujson.load(codecs.open(word_idx_file, encoding='utf8'))

# Read and transform the test set
X_test = testset_read(test_fn, word_idx, 25)

# Get the predictions for the test set
predictions = model.predict(X_test)

for pred in predictions:
    print pred[0]

