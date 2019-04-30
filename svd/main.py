import argparse
import math
import os
import pickle
from random import randrange
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import dok_matrix, save_npz, load_npz
from sklearn.decomposition import TruncatedSVD

sys.path.append("..")
from utils import load_corpus_and_vocab, create_corpus_and_vocab

def create_co_matrix(corpus, vocab):
    """ Creates the co-occurrence matrix of a corpus.

        Returns a scipy.sparse.csr_matrix where matrix[i, j]
        is the frequency that words with vocab index i and j
        occur in each others" contexts.

        Note that the context window size is sampled ~Unif(1, 10).
        This has the same effect as harmonic weighting.
    """

    # Sets size and creates matrix.
    size = len(vocab)
    matrix = dok_matrix((size, size), np.dtype(int))

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
            context_words = pre_context + post_context

            # Updates co-occurrence values in the matrix.
            for context_word in context_words:
                matrix[vocab[anchor_word], vocab[context_word]] += 1

    # Converts co-occurrence matrix to csr_matrix format and saves it as a .npz file.
    matrix = matrix.tocsr()
    save_npz(FLAGS.co_matrix_path, matrix)
    print("Saved co-occurrence matrix.")

    return matrix

def create_ppmi_matrix(co_matrix):
    """ Creates the positive pointwise mutual information (PPMI)
        matrix from a co-occurrence matrix.

        Returns a scipy.sparse.csr_matrix where matrix[i, j]
        is the PPMI of words with vocab index i and j.

        PPMI = max( Pr[i U j] / (Pr[i] * Pr[j]), 0 )
    """

    # Converts co-occurrence matrix to dok_matrix
    # format for easier indexing.
    co_matrix = co_matrix.todok()

    # Sets size and creates matrix.
    size = co_matrix.shape[0]
    ppmi_matrix = dok_matrix((size, size))

    # Computes total, row, and column sums.
    co_total = co_matrix.sum()
    co_row_sum = co_matrix.sum(axis=1)
    co_col_sum = co_matrix.sum(axis=0).transpose()

    # Iterates over all nonzero entries in the co-occurrence matrix.
    nonzero = co_matrix.nonzero()
    for entry_index, (anchor_word_index, context_word_index) in enumerate(zip(*nonzero)):
        # Prints progress
        if entry_index % 1e5 == 0:
            print("Processing entry {:.2e}/{:.2e}".format(entry_index, len(nonzero[0])))

        # Gets total occurrences of each word.
        anchor_word_sum = co_row_sum.item(anchor_word_index)
        context_word_sum = co_col_sum.item(context_word_index)

        # Calculates probabilities.
        joint_prob = co_matrix[anchor_word_index, context_word_index] / co_total
        anchor_word_marginal_prob = anchor_word_sum / co_total
        context_word_marginal_prob = context_word_sum / co_total

        # Calculates PPMI.
        ratio = joint_prob / (anchor_word_marginal_prob * context_word_marginal_prob)
        pmi = math.log(ratio)
        ppmi = max(pmi, 0)

        # Updates PPMI values in the matrix.
        ppmi_matrix[anchor_word_index, context_word_index] = ppmi

    # Converts PPMI matrix to csr_matrix format and saves it as a .npz file.
    ppmi_matrix = ppmi_matrix.tocsr()
    save_npz(FLAGS.ppmi_matrix_path, ppmi_matrix)
    print("Saved PPMI matrix.")

    return ppmi_matrix

def get_vocab_and_ppmi_matrix():
    """ Loads and/or creates the corpus and vocab,
        co-occurrence matrix, and the PPMI matrix.

        Returns the vocab and PPMI matrix.
    """

    # Loads the vocab and PPMI matrix if possible.
    if os.path.isfile(FLAGS.vocab_path) and os.path.isfile(FLAGS.ppmi_matrix_path):
        vocab = pickle.load(open(FLAGS.vocab_path, "rb"))
        ppmi_matrix = load_npz(FLAGS.ppmi_matrix_path)
        print("Loaded vocab and PPMI matrix.\n")

    # Creates the PPMI matrix.
    else:
        # Load the co-occurrence matrix if possible.
        if os.path.isfile(FLAGS.co_matrix_path):
            co_matrix = load_npz(FLAGS.co_matrix_path)
            print("Loaded co-occurrence matrix.\n")

        # Creates the co-occurrence matrix.
        else:
            # Loads the corpus and vocab if possible.
            if os.path.isfile(FLAGS.corpus_path) and os.path.isfile(FLAGS.vocab_path):
                corpus, vocab = load_corpus_and_vocab(FLAGS.corpus_path,
                                                      FLAGS.vocab_path,)
                print("Loaded corpus and vocab.\n")

            # Creates the corpus and vocab.
            else:
                print("Creating corpus and vocab...\n")
                corpus, vocab = create_corpus_and_vocab(FLAGS.data_path,
                                                        FLAGS.corpus_path, FLAGS.vocab_path,
                                                        FLAGS.num_articles, FLAGS.min_freq)
                print("Done creating corpus and vocab.")
            print("Length of vocab: {}\n".format(len(vocab)))

            # Creates the co-occurrence matrix.
            print("Creating co-occurrence matrix...\n")
            co_matrix = create_co_matrix(corpus, vocab)
            print("Done creating co-occurrence matrix.\n")

        # Creates the PPMI matrix.
        print("Creating PPMI matrix...\n")
        ppmi_matrix = create_ppmi_matrix(co_matrix)
        print("Done creating PPMI matrix.\n")

    return vocab, ppmi_matrix

def dist(word_vec1, word_vec2):
    """ Returns the distance between two word vectors. """

    # Calculates vector distance via the norm of the difference.
    return np.linalg.norm(word_vec1 - word_vec2)

def n_closest_words(word, n, reduced_matrix, vocab, reversed_vocab):
    """ Returns the n closest words to a given target word. """

    # Gets the word vector for the target word.
    word_row = reduced_matrix[vocab[word]]

    # Calculates the distance between the target word and all other words.
    dists = np.asarray([dist(word_row, row) for row in reduced_matrix])

    # Selects the n indices with minimal distance.
    top_n_indices = np.argpartition(dists, n + 1)[:n + 1]

    # Transforms the indices into their corresponding words.
    top_n_words = [reversed_vocab[index] for index in top_n_indices if index != vocab[word]]

    return top_n_words

def n_dim_reduced_matrix(n_dim, matrix):
    """ Reduces dimensionality of the PPMI matrix via SVD. """

    # Instantiates an SVD and reduces the PPMI matrix.
    svd = TruncatedSVD(n_components=n_dim, random_state=1)
    svd.fit(matrix)
    reduced_matrix = svd.transform(matrix)

    return svd, reduced_matrix

def main():
    vocab, ppmi_matrix = get_vocab_and_ppmi_matrix()
    _, reduced_matrix = n_dim_reduced_matrix(80, ppmi_matrix)

    reversed_vocab = dict(zip(vocab.values(), vocab.keys()))

    top30_words_men = n_closest_words("men", 30, reduced_matrix, vocab, reversed_vocab)
    top30_words_women = n_closest_words("women", 30, reduced_matrix, vocab, reversed_vocab)

    print("Top 30 Words Associated with MEN: " + str(top30_words_men))
    print("Top 30 Words Associated with WOMEN: " + str(top30_words_women))

#    x = np.arange(2, 222)
#    y = np.zeros(x.size)
#    for ind, dim in enumerate(x):
#        print("Calculating {}-dimension reduction".format(dim))
#        svd, _ = n_dim_reduced_matrix(dim, ppmi_matrix)
#        y[ind] = svd.explained_variance_ratio_.sum()
#
#    fig = plt.figure()
#    ax =  plt.axes()
#    ax.plot(x, y)
#    plt.savefig("variance.png")

#    men_row = reduced_matrix[vocab["men"]]
#    women_row = reduced_matrix[vocab["women"]]
#    leader_row = reduced_matrix[vocab["leader"]]
#    leaders_row = reduced_matrix[vocab["leaders"]]
#    print(dist(men_row, leader_row), dist(women_row, leader_row))
#    print(dist(men_row, leaders_row), dist(women_row, leaders_row))
#    vis_matrix = n_dim_reduced_matrix(2, ppmi_matrix)
#    men = vis_matrix[vocab["men"]]
#    women = vis_matrix[vocab["women"]]
#    leader = vis_matrix[vocab["leader"]]
#    engineer = vis_matrix[vocab["engineer"]]
#    raped = vis_matrix[vocab["raped"]]
#    genital = vis_matrix[vocab["genital"]]
#
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    x = [men[0], women[0], leader[0], engineer[0], raped[0], genital[0]]
#    y = [men[1], women[1], leader[1], engineer[1], raped[1], genital[1]]
#    colors = ("red", "blue", "green", "brown", "black", "purple")
#    labels = ("men", "women", "leader", "engineer", "raped", "genital")
#
#    for x, y, color, label in zip(x, y, colors, labels):
#        ax.scatter(x, y, c=color, label=label)
#    plt.legend()
#    plt.savefig("test.png")


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

    parser.add_argument("--co_matrix_path", default="../saved/svd/co_matrix.npz", type=str,
                        help="Path to co-occurrence matrix saved as a .npz file \
                              (default: ../saved/svd/co_matrix.npz)")

    parser.add_argument("--ppmi_matrix_path", default="../saved/svd/ppmi_matrix.npz", type=str,
                        help="Path to PPMI matrix saved as a .npz file \
                              (default: ../saved/svd/ppmi_matrix.npz)")

    # Sets arguments as FLAGS
    FLAGS, _ = parser.parse_known_args()

    main()
