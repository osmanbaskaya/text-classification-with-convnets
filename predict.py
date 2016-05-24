import sys
import ujson
import codecs
from keras.models import model_from_json
from data import testset_read

test_fn = sys.argv[1]
model_weight_file = sys.argv[2]
arch_file = model_weight_file + '-arch.json'
word_idx_file = model_weight_file + '-word_idx.json'

# TODO: Put these in a config file.
batch_size = 32
maxlen = 25


model = model_from_json(open(arch_file).read())
model.load_weights(model_weight_file)

word_idx = ujson.load(codecs.open(word_idx_file, encoding='utf8'))

X_test = testset_read(test_fn, word_idx, 25)

model.predict(X_test)
