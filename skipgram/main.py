import argparse
import math
import os
import pickle
from random import randrange
import sys

from keras.layers import Dense, Activation
from keras.models import Sequential
from mpu.ml import indices2one_hot
import numpy as np

sys.path.append("..")
from utils import load_corpus_and_vocab, create_corpus_and_vocab

def create_train_set(corpus, vocab):
    """ Creates the training set.

        Returns a tuple of lists where the first is the anchor
        word indices and the second is the context word indices.

        Note that the context window size is sampled ~Unif(1, 10).
        This has the same effect as harmonic weighting.
    """

    anchor_words = []
    context_words = []

    # Iterates through every article in the corpus
    for article_index, article in enumerate(corpus):
        # Prints progress.
        if article_index % 100 == 0:
            print("Processing article {}/{}".format(article_index, FLAGS.num_articles))

        # Converts article to list of words for processing.
        article = article.split()
        for anchor_word_index, anchor_word in enumerate(article):
            # Samples window size.
            window_size = randrange(1, 11)

            # Obtains context words.
            pre_context = article[max(anchor_word_index - window_size, 0) : \
                                  anchor_word_index]
            post_context = article[min(anchor_word_index + 1, len(article)) : \
                                   min(anchor_word_index + window_size + 1, len(article))]
            context = pre_context + post_context

            # Update training set values
            for context_word in context:
                anchor_words.append(vocab[anchor_word])
                context_words.append(vocab[context_word])

    assert len(anchor_words) == len(context_words)

    # Saves training set as a .pickle file
    train = (anchor_words, context_words)
    pickle.dump(train, open(FLAGS.train_path, "wb"))
    return train

def get_vocab_and_train_set():
    """ Loads and/or creates the corpus and vocab
        and the training set (anchor words and context words).

        Returns the vocab and training set.
    """

    # Loads the corpus and vocab if possible.
    if os.path.isfile(FLAGS.corpus_path) and os.path.isfile(FLAGS.vocab_path):
        corpus, vocab = load_corpus_and_vocab(FLAGS.corpus_path,
                                              FLAGS.vocab_path,)
        print("Loaded corpus and vocab.")

    # Creates the corpus and vocab.
    else:
        print("Creating corpus and vocab...\n")
        corpus, vocab = create_corpus_and_vocab(FLAGS.data_path,
                                                FLAGS.corpus_path, FLAGS.vocab_path,
                                                FLAGS.num_articles, FLAGS.min_freq)
        print("Done creating corpus and vocab.")

    print("Length of vocab: {}\n".format(len(vocab)))

    # Loads the training set if possible.
    if os.path.isfile(FLAGS.train_path):
        train = pickle.load(open(FLAGS.train_path, "rb"))
        print("Loaded training set.")

    # Creates the training set.
    else:
        print("Creating training set...\n")
        train = create_train_set(corpus, vocab)
        print("Done creating training set.")

    print("Length of training set: {}\n".format(len(train[0])))

    return vocab, train

def create_model(net_size, input_size):
    """ Defines a one-layer Keras model with
        softmax activation and categorical loss.
    """

    model = Sequential([
        Dense(net_size, input_shape=(input_size,)),
        Dense(input_size),
        Activation("softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model

def train_model(model, train, vocab_size):
    """ Trains the model. """

    # Calculates the number of batches per epoch
    num_batches = math.ceil(len(train[0]) / FLAGS.batch_size)

    # Halves the dataset into data and labels
    anchors = train[0]
    contexts = train[1]

    # Trains the model for the specified number of epochs
    for epoch in range(1, FLAGS.epochs + 1):
        # Creates batches of data for training
        for batch_num in range(num_batches):
            # Gets the batch from the dataset
            batch_start = FLAGS.batch_size * batch_num
            batch_end = FLAGS.batch_size * (batch_num + 1)
            anchors_batch = anchors[batch_start : min(batch_end, len(anchors))]
            contexts_batch = contexts[batch_start : min(batch_end, len(contexts))]

            # Converts batch to one hot vectors
            anchors_batch = indices2one_hot(anchors_batch, vocab_size)
            contexts_batch = indices2one_hot(contexts_batch, vocab_size)

            # Converts batch to np arrays
            anchors_batch = np.asarray(anchors_batch)
            contexts_batch = np.asarray(contexts_batch)

            # Fits model on batch
            model.fit(anchors_batch, contexts_batch)

        # Prints progress and saves weights
        print("----EPOCH {} COMPLETED----".format(epoch))
        model.save_weights("../saved/skipgram/weights/weights{}.h5".format(epoch))

    return model

def main():
    vocab, train = get_vocab_and_train_set()
    model = create_model(FLAGS.net_size, input_size=len(vocab))
    model = train_model(model, train, vocab_size=len(vocab))

if __name__ == "__main__":
    # Instantiates an argument parser
    parser = argparse.ArgumentParser()

    # Adds command line arguments
    parser.add_argument("--min_freq", default=182, type=int,
                        help="Minimum frequency a word needs to remain in \
                              the corpus during preprocessing \
                              (default: 182, which gives the top 10,029 words \
                              in the 50,000 articles case)")

    parser.add_argument("--num_articles", default=50000, type=int,
                        help="Number of articles to process \
                              (default: 50,000, max: 50,000)")

    parser.add_argument("--data_path", default="/hdd0/datasets/fairness/articles.csv",
                        type=str, help="Path to dataset \
                                        (default: /hdd0/datasets/fairness/articles.csv)")

    parser.add_argument("--corpus_path", default="../saved/corpus.pickle", type=str,
                        help="Path to vocab saved as a .pickle file \
                              (default: saved/corpus.pickle)")

    parser.add_argument("--vocab_path", default="../saved/vocab.pickle", type=str,
                        help="Path to vocab saved as a .pickle file \
                              (default: ../saved/vocab.pickle)")

    parser.add_argument("--train_path", default="../saved/skipgram/train.pickle", type=str,
                        help="Path to training set saved as a .pickle file \
                              (default: ../saved/skipgram/train.pickle)")

    parser.add_argument("--net_size", default=32, type=int,
                        help="Neural network size (default: 32)")

    parser.add_argument("--epochs", default=10, type=int,
                        help="Num of epochs for training (default: 10)")

    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size for training (default: 128)")

    # Sets arguments as FLAGS
    FLAGS, _ = parser.parse_known_args()

    main()
