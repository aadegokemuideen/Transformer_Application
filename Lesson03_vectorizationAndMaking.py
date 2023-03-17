##find out the statistics from the mode;
import pickle
import random
import tensorflow
import tensorflow as tf


from tensorflow.keras.layers import TextVectorization


# load normalized sentence pairs
with open("text_pairs.pickle", "rb") as fp:
	text_pairs = pickle.load(fp)

#train_test_val; split of randomized sentence pairs
random.shuffle(text_pairs)
n_val = int(0.15 * len(text_pairs))
n_train = len(text_pairs) -2 * n_val
train_pairs = text_pairs[:n_train]
val_pairs  = text_pairs[n_train:n_train+n_val]
test_pairs = text_pairs[n_train+n_val:]


# Parameter deterimine after analyzing the input data
vocab_size_en = 10000
vocal_size_fr = 20000
seq_length    = 20

# Create vectorizer
eng_vectorizer  = TextVectorization(
    max_tokens  = vocab_size_en,
    standardize = None,
    split = "whitespace",
    output_mode = "int",
    output_sequence_length = seq_length
	)

fra_vectorizer = TextVectorization(
    max_tokens = vocal_size_fr,
    standardize = None,
    split = "whitespace",
    output_mode = "int",
    output_sequence_length = seq_length + 1
	)

# train the vectorization layer using training data set
train_eng_texts = [pairs[0] for pairs in train_pairs]
train_fra_texts = [pairs[1] for pairs in train_pairs]
eng_vectorizer.adapt(train_eng_texts)
fra_vectorizer.adapt(train_fra_texts)


# save for subsequence steps
with open("vectorize.pickle", "wb") as fp:

	data = {
	  "train": train_pairs,
	  "val"  : val_pairs,
	  "test" : test_pairs,
	  "engvec_config" : eng_vectorizer.get_config(),
	  "engvec_weights": eng_vectorizer.get_weights(),
	  "fravec_config" : fra_vectorizer.get_config(),
	  "fravec_weights": fra_vectorizer.get_weights()
	}

	pickle.dump(data, fp)



# load text data and vectorizer weights
with open("vectorize.pickle", "rb") as fp:
	data = pickle.load(fp)


train_pairs = data["train"]
val_pairs   = data["val"]
test_pairs  = data["test"]  # not used

eng_vectorizer = TextVectorization.from_config(data["engvec_config"])
eng_vectorizer.set_weights(data["engvec_weights"])
fra_vectorizer = TextVectorization.from_config(data["fravec_config"])
fra_vectorizer.set_weights(data["fravec_weights"])


## Let set up the data set object
"""Take an English and a French sentence pair, convert into input and target.
    The input is a dict with keys `encoder_inputs` and `decoder_inputs`, each
    is a vector, corresponding to English and French sentences respectively.
    The target is also vector of the French sentence, advanced by 1 token. All
    vector are in the same length.
 
    The output will be used for training the transformer model. In the model we
    will create, the input tensors are named `encoder_inputs` and `decoder_inputs`
    which should be matched to the keys in the dictionary for the source part
"""
def format_dataset(eng, fra): 

    eng = eng_vectorizer(eng)
    fra = fra_vectorizer(fra)

    source = {"encoder_inputs": eng,
              "decoder_inputs": fra[:, :-1]}
    target  = fra[:,1:]

    return (source, target)



""" Create Tensorflow Dataset for the sentence pairs"""
def make_dataset(pairs, batch_size=64):
    # aggregate sentencees using zip(*pairs)
 	eng_texts, fra_texts = zip(*pairs)
 	# convert them into list, and the create tensors
 	dataset = tf.data.Dataset.from_tensor_slices((list(eng_texts),list(fra_texts)))
 	return dataset.shuffle(2048) \
 	               .batch(batch_size).map(format_dataset) \
 	               .prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds   = make_dataset(val_pairs)


# test the dataset
for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["encoder_inputs"][0]: {inputs["encoder_inputs"][0]}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"][0]: {inputs["decoder_inputs"][0]}')
    print(f"targets.shape: {targets.shape}")
    print(f"targets[0]: {targets[0]}")

