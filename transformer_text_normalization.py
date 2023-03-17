## Text Normalization
import pathlib
import pickle
import random
import re
import unicodedata
import matplotlib.pyplot as plt

#import tensorflow as tf




def normalize(line):
	""" Normalize a line of text and split into two at the table caharacter"""
	line = unicodedata.normalize("NFKC", line.strip().lower())
	line = re.sub(r"^([^ \w])(?!\s)", r"\1 ",line)
	line = re.sub(r"(\s[^ \w])(?!\s)", r"\1", line)
	line = re.sub(r"(?!\s)(^ \w)$", r"\1", line)
	line = re.sub(r"(?!\s)([^ \w]\s)", r" \1", line)

	eng, fra = line.split("\t")
	fra = "[start] " + fra + " [end]"

	return eng, fra


# normalize each line and separate into English and French
with open("./dataset/fra.txt") as f:
	text_pairs = [normalize(line) for line in f]
	#textfile = f.read()  # f.readline()

#print some samples
for _ in range(5):
	print(random.choice(text_pairs))

with open("text_pairs.pickle", "wb") as fp:
	pickle.dump(text_pairs, fp)



with open("text_pairs.pickle", "rb") as fp:
	text_pairs = pickle.load(fp)


#count tokens
eng_tokens, fra_tokens = set(), set()
eng_maxlen, fra_maxlen = 0, 0 
for eng, fra in text_pairs:
	eng_tok, fra_tok = eng.split(), fra.split()
	eng_maxlen = max(eng_maxlen, len(eng_tok))
	fra_maxlen = max(fra_maxlen, len(fra_tok))
	eng_tokens.update(eng_tok)
	fra_tokens.update(fra_tok)


print(f"Total English tokens: {len(eng_tokens)}")
print(f"Total French tokens {len(fra_tokens)}")
print(f"Max English length: {eng_maxlen}")
print(f"Max Frence length: {fra_maxlen}")
print(f"{len(text_pairs)} total pairs")


##Check for the distiribution of the data
with open("text_pairs.pickle", "rb") as fp:
	text_pairs = pickle.load(fp)

# Histogram of sentence length in tokens
en_lengths = [len(eng.split()) for eng, fra in text_pairs]
fr_lengths = [len(fra.split()) for eng, fra in text_pairs]


plt.hist(en_lengths, label="en", color="red", alpha=0.33)
plt.hist(fr_lengths, label="fr", color = "blue", alpha =0.33)
plt.yscale("log") # sentence length fits Benfords's law
plt.ylim(plt.ylim()) # make y-axis consistent for both plots
plt.plot([max(en_lengths), max(en_lengths)], plt.ylim(), color="red")
plt.plot([max(fr_lengths), max(fr_lengths)], plt.ylim(), color="blue")
plt.legend()
plt.title("Example count vs Token length")
plt.show()
