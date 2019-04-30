# Read the dataset into a Pandas DataFrame.
# Print and check the first few entries with the iloc function.

import pandas
df = pandas.read_csv("/hdd0/datasets/fairness/articles.csv", engine="python",
                     encoding="utf-8", error_bad_lines=False)

# Perform whatever dataset exploration you wish.

# Retrieve the first NUM_ARTICLES entries in the content column and
# turn it into a list. Print and check the first few entries.

NUM_ARTICLES = 50000
corpus = list(df["content"])[:NUM_ARTICLES]

# Dataset cleanup (USE LIST COMPREHENSION):
# (1) Separate the data into a list of sentences
# (2) Convert the sentences to lowercase
# (3) Remove all punctuation (watch out for chars like £)
# (4) Remove all numbers
# (5) Remove excessive spaces
# (6) Separate the data into a list of sentences containing lists of words
# (7) Remove all sentences with less than 4 words
#
# Example:
# [['foo', 'bar'], ['bar', 'foo']]
#
# Print and check the number of sentences and the first few entries.

import string
import re

sentences = [article.split(".") for article in corpus]
sentences = [sentence[:-1] for sentence in sentences]
sentences = [sentence.lower() for article in sentences for sentence in article]
sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in sentences]
sentences = [sentence.replace('“', '').replace('”', '').replace('’', '\'').replace('£', '').replace('—', '') for sentence in sentences]
sentences = [re.sub(r'\d+', '', sentence) for sentence in sentences]
sentences = [" ".join(sentence.split()) for sentence in sentences]
sentences = [re.sub("[^\w]", " ",  sentence).split() for sentence in sentences]
sentences = [sentence for sentence in sentences if len(sentence) > 3]
print("Num of sentences: {0}".format(len(sentences)))

# Create a dictionary of word : frequency
counts = {}
for sentence in sentences:
    for word in sentence:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

# Graph word frequencies in a bar chart using pyplot
import matplotlib.pyplot as plt
frequencies = {}
for word in counts:
  if counts[word] in frequencies:
    frequencies[counts[word]] += 1
  else:
    frequencies[counts[word]] = 1

plt.bar(list(frequencies.keys()), list(frequencies.values()))
plt.xlim((1, 100))
plt.savefig("word_frequencies.png")

# Replace all words which occur at a low frequency with 
# a '?' character. Use your graph to determine the cutoff.
# This heavily reduces the dimensionality of the dataset 
# at a cost of not being able to map rare words.
#
# At the same time, create the vocab:
# the dict of all words with no duplicates.
# Make the value of the word the length of the vocab at the
# time the word was added -- this is to give each word a
# unique index accessible in O(1) time.
#
# Example:
# {'foo': 0, 'bar': 1, '?': 2}
#
# Print and check the length.

vocab = {}
for sentence in sentences:
  for word in sentence:
    if counts[word] < 151:
      sentence[sentence.index(word)] = "?"
    elif word not in vocab:
      vocab.setdefault(word, len(vocab))

vocab.setdefault("?", len(vocab))
print("Length of vocab: {0}".format(len(vocab)))

# We're trying to predict contextual words from an input word (skip-gram model).
# We have to first extract the contextual words.
# Use a window size of 2 (on both sides) and create two lists, where
# one is the current word and the other a contextual word within the window size.
#
# Example:
# x = ['foo', 'foo', 'bar']
# y = ['bar', 'bar2', 'foo']
#
# But don't save it like that, because it'll use too much RAM! Instead,
# immediately convert to the index of the word in the vocabulary.
#
# Example:
# x = [1, 1, 2]
# y = [2, 3, 1]

WINDOW_SIZE = 3

x = []
y = []
for sentence_index, sentence in enumerate(sentences):
  if sentence_index % 10000 == 0:
    print("Processing sentence {0}".format(sentence_index))
  for word_index, word in enumerate(sentence):
      if word is not "?":
        pre_context = sentence[max(word_index - WINDOW_SIZE, 0) : word_index]
        post_context = sentence[min(word_index + 1, len(sentence)) : min(word_index + WINDOW_SIZE + 1, len(sentence))]
        context_words = pre_context + post_context

        for context_word in context_words:
            x.append(vocab[word])
            y.append(vocab[context_word])

print("Length of training set: {0}".format(len(x)))

# Save x and y and vocab with pickle

import pickle

x_pickle = open('x.pickle', 'wb')
y_pickle = open('y.pickle', 'wb')
vocab_pickle = open('vocab.pickle', 'wb')
pickle.dump(x, x_pickle)
pickle.dump(y, y_pickle)
pickle.dump(vocab, vocab_pickle)
x_pickle.close()
y_pickle.close()
vocab_pickle.close()

# Define a to_one_hot function which takes an index
# and returns a list of all zeros except a 1 in that index.
#
# Also define training hyperparameters: EPOCHS,  BATCH_SIZE, NUM_BATCHES

import numpy as np
import math

def to_one_hot(word_index):
  vector = [0] * len(vocab)
  vector[word_index] = 1
  return vector

EPOCHS = 1
BATCH_SIZE = 4096
NUM_BATCHES = math.ceil(len(x) / BATCH_SIZE)

# Define a one-layer Keras model with softmax activation and categorical loss
# Print and check the model summary

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(len(vocab),)),
    Dense(len(vocab)),
    Activation('softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model!
# You'll need to get BATCH_SIZE sized slices of x and y,
# then transform them to one hot vectors. Convert
# those to nparrays and use the fit() method to train.

for epoch in range(1, EPOCHS + 1):
  for i in range(NUM_BATCHES):
    batch_start = BATCH_SIZE * i
    batch_end = BATCH_SIZE * (i + 1)
    x_batch = x[batch_start : min(batch_end, len(x))]
    y_batch = y[batch_start : min(batch_end, len(y))]
    
    for j, _ in enumerate(x_batch):
      x_batch[j] = to_one_hot(x_batch[j])
      y_batch[j] = to_one_hot(y_batch[j])

    x_batch = np.asarray(x_batch)
    y_batch = np.asarray(y_batch)

    model.fit(x_batch, y_batch)
    
  print("----EPOCH {0} COMPLETED----".format(epoch))
  model.save_weights("weights.h5")
