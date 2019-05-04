import argparse
import math
import os
import pickle
from random import randrange
import sys

from keras.callbacks import ModelCheckpoint
import numpy as np
from scipy.sparse import load_npz

sys.path.append("..")
from model import create_model
from utils import load_corpus_and_vocab, create_corpus_and_vocab, create_co_matrix

def create_negative_sampling_weights(corpus, vocab):
    """ Creates a list of negative sampling weights, where weights[i] is the
        probability that the word with index i will be selected.

        Negative samples are sampled according to the smoothed unigram
        distribution: Pr[c] = (#c)^a / sum[(#w)^a]. Levy and Goldberg 2015
        suggest a=0.75.
    """

    # Load the co-occurrence matrix if possible.
    if os.path.isfile(FLAGS.co_matrix_path):
        co_matrix = load_npz(FLAGS.co_matrix_path)
        print("Loaded co-occurrence matrix.\n")

    # Creates the co-occurrence matrix.
    else:
        print("Creating co-occurrence matrix...\n")
        co_matrix = create_co_matrix(corpus, vocab, FLAGS.num_articles,
                                     FLAGS.co_matrix_path)
        print("Done creating co-occurrence matrix.\n")

    # Computes total and row sums with smoothing.
    co_row_sum = co_matrix.sum(axis=1)
    co_row_sum_smoothed = [math.pow(count[0], 0.75) for count in co_row_sum]
    co_total_smoothed = np.sum(co_row_sum_smoothed)

    # Creates the weights list.
    weights = [count / co_total_smoothed for count in co_row_sum_smoothed]

    # Ensures weights are correct and saves them as a .pickle file.
    assert abs(np.sum(weights) - 1.) < 0.01
    pickle.dump(weights, open(FLAGS.negative_weights_path, "wb"))

    return weights

def create_data(corpus, vocab):
    """ Creates the dataset.

        Returns a list of the form [((anchor_word, context_word), label)]
        where the label tells whether the context word is a real context word
        or a negatively sampled one.

        Note that the context window size is sampled ~Unif(1, 10).
        This has the same effect as harmonic weighting.
    """

    # Loads the negative sampling weights if possible.
    if os.path.isfile(FLAGS.negative_weights_path):
        weights = pickle.load(open(FLAGS.negative_weights_path, "rb"))
        print("Loaded negative sampling weights.\n")

    # Creates the negative sampling weights.
    else:
        print("Creating negative sampling weights...\n")
        weights = create_negative_sampling_weights(corpus, vocab)
        print("Done creating negative sampling weights.\n")

    anchor_indices = []
    context_indices = []
    labels = []

    # Creates dataset from corpus.
    for article_index, article in enumerate(corpus):
        # Prints progress.
        if article_index % 100 == 0:
            print("Processing article {:,}/{:,}".format(article_index, FLAGS.num_articles))

        # Converts article to list of words for processing.
        article = article.split()
        for anchor_word_index_in_article, anchor_word in enumerate(article):
            # Samples window size.
            window_size = randrange(1, 11)

            # Obtains context words.
            pre_context = article[max(anchor_word_index_in_article - window_size, 0) : \
                                  anchor_word_index_in_article]
            post_context = article[min(anchor_word_index_in_article + 1, len(article)) : \
                                   min(anchor_word_index_in_article + window_size + 1, len(article))]
            context_words = pre_context + post_context

            # Updates dataset values.
            for context_word in context_words:
                anchor_indices.append(vocab[anchor_word])
                context_indices.append(vocab[context_word])
                labels.append(1)

            # Samples negative words.
            negative_word_indices = np.random.choice(a=len(vocab),
                                                     size=FLAGS.num_negative_samples,
                                                     replace=False,
                                                     p=weights)

            # Updates dataset values.
            for negative_word_index in negative_word_indices:
                anchor_indices.append(vocab[anchor_word])
                context_indices.append(negative_word_index)
                labels.append(0)

    # Saves dataset as .pickle files.
    os.makedirs(FLAGS.anchor_indices_path, exist_ok=True)

    pickle.dump(anchor_indices, open(FLAGS.anchor_indices_path, "wb"))
    pickle.dump(context_indices, open(FLAGS.context_indices_path, "wb"))
    pickle.dump(labels, open(FLAGS.labels_path, "wb"))

    return anchor_indices, context_indices, labels

def get_vocab_and_data():
    """ Loads and/or creates the corpus and vocab
        and the data (anchor words, context words, and label).

        Returns the vocab and data.
    """

    # Loads the corpus and vocab if possible.
    if os.path.isfile(FLAGS.corpus_path) and os.path.isfile(FLAGS.vocab_path):
        corpus, vocab = load_corpus_and_vocab(FLAGS.corpus_path,
                                              FLAGS.vocab_path,)
        print("Loaded corpus and vocab.")

    # Creates the corpus and vocab.
    else:
        print("Creating corpus and vocab...\n")
        corpus, vocab = create_corpus_and_vocab(FLAGS.csv_path,
                                                FLAGS.corpus_path, FLAGS.vocab_path,
                                                FLAGS.num_articles, FLAGS.min_freq)
        print("Done creating corpus and vocab.")

    print("Length of vocab: {:,}\n".format(len(vocab)))

    # Loads the dataset if possible.
    if os.path.isfile(FLAGS.anchor_indices_path) and os.path.isfile(FLAGS.context_indices_path) and os.path.isfile(FLAGS.labels_path):
        anchor_indices = pickle.load(open(FLAGS.anchor_indices_path), "rb")
        context_indices = pickle.load(open(FLAGS.context_indices_path), "rb")
        labels = pickle.load(open(FLAGS.labels_path), "rb")
        print("Loaded data.")

    # Creates the dataset.
    else:
        print("Creating data...\n")
        anchor_indices, context_indices, labels = create_data(corpus, vocab)
        print("Done creating data.")

    print("Length of data: {:,}\n".format(len(data)))

    # Converts dataset to np arrays.
    anchor_indices = np.asarray(anchor_indices)
    context_indices = np.asarray(context_indices)
    labels = np.asarray(labels)

    return vocab, anchor_indices, context_indices, labels

def train_model(model, anchor_indices, context_indices, labels):
    """ Trains the model. """

    # Creates input for model.
    input_pairs = {"anchor_index": anchor_indices,
                   "context_index": context_indices}

    # Creates checkpoint callback
    checkpoint = ModelCheckpoint(filepath=FLAGS.weights_path,
                                 save_weights_only=True)

    # Trains the model
    model.fit(x=input_pairs, y=labels, validation_split=0.2, epochs=FLAGS.epochs,
              batch_size=FLAGS.batch_size, callbacks=[checkpoint])

    # # Calculates the number of batches per epoch
    # num_batches = math.ceil(len(data) / FLAGS.batch_size)

    # # Trains the model for the specified number of epochs
    # for epoch in range(1, FLAGS.epochs + 1):
    #     # Creates batches of data for training
    #     for batch_num in range(num_batches):
    #         # Gets the batch from the dataset
    #         batch_start = FLAGS.batch_size * batch_num
    #         batch_end = FLAGS.batch_size * (batch_num + 1)
    #         anchors_batch = anchors[batch_start : min(batch_end, len(anchors))]
    #         contexts_batch = contexts[batch_start : min(batch_end, len(contexts))]

    #         # Converts batch to one hot vectors
    #         anchors_batch = indices2one_hot(anchors_batch, vocab_size)
    #         contexts_batch = indices2one_hot(contexts_batch, vocab_size)

    #         # Converts batch to np arrays
    #         anchors_batch = np.asarray(anchors_batch)
    #         contexts_batch = np.asarray(contexts_batch)

    #         # Fits model on batch
    #         model.fit(anchors_batch, contexts_batch)

    #     # Prints progress and saves weights
    #     print("----EPOCH {} COMPLETED----".format(epoch))
    #     model.save_weights(FLAGS.weights_folder + "/weights{}.h5".format(epoch))

    return model

def main():
    vocab, anchor_indices, context_indices, labels = get_vocab_and_data()
    model = create_model(vector_dim=FLAGS.vector_dim, vocab_size=len(vocab))
    model = train_model(model, anchor_indices, context_indices, labels)

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

    parser.add_argument("--csv_path", default="/hdd0/datasets/fairness/articles.csv",
                        type=str, help="Path to dataset \
                                        (default: /hdd0/datasets/fairness/articles.csv)")

    parser.add_argument("--corpus_path", default="../saved/corpus.pickle",
                        type=str, help="Path to vocab saved as a .pickle file \
                                        (default: saved/corpus.pickle)")

    parser.add_argument("--vocab_path", default="../saved/vocab.pickle",
                        type=str, help="Path to vocab saved as a .pickle file \
                                        (default: ../saved/vocab.pickle)")

    parser.add_argument("--co_matrix_path", default="../saved/co_matrix.npz",
                        type=str, help="Path to co-occurrence matrix saved as \
                                        a .npz file (default: ../saved/co_matrix.npz)")

    parser.add_argument("--negative_weights_path",
                        default="../saved/skipgram/negative_weights.pickle",
                        type=str, help="Path to negative sampling weights list \
                                        saved as a .pickle file \
                                        (default: ../saved/skipgram/\
                                        negative_weights.pickle)")

    parser.add_argument("--anchor_indices_path", default="../saved/skipgram/data/anchor_indices.pickle",
                        type=str, help="Path to anchor indices saved as a .pickle file \
                                        (default: ../saved/skipgram/data/anchor_indices.pickle)")

    parser.add_argument("--context_indices_path", default="../saved/skipgram/data/context_indices.pickle",
                        type=str, help="Path to context indices saved as a .pickle file \
                                        (default: ../saved/skipgram/data/context_indices.pickle)")

    parser.add_argument("--labels_path", default="../saved/skipgram/data/labels.pickle",
                        type=str, help="Path to labels saved as a .pickle file \
                                        (default: ../saved/skipgram/data/labels.pickle)")

    parser.add_argument("--weights_path",
                        default="../saved/skipgram/weights/weights-{epoch:02d}-{val_acc:.2f}.hdf5",
                        type=str, help="Path to network weights folder \
                                        (default: ../saved/skipgram/weights/\
                                        weights-{epoch:02d}-{val_acc:.2f}.hdf5)")

    parser.add_argument("--vector_dim", default=100, type=int,
                        help="Embedding vector size (default: 100)")

    parser.add_argument("--num_negative_samples", default=5, type=int,
                        help="Number of negative samples per occurrence of \
                              a positive sample in the data (default: 5)")

    parser.add_argument("--epochs", default=10, type=int,
                        help="Num of epochs for training (default: 10)")

    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size for training (default: 128)")

    # Sets arguments as FLAGS
    FLAGS, _ = parser.parse_known_args()

    main()
