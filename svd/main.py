import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import dok_matrix, save_npz, load_npz
from sklearn.decomposition import TruncatedSVD

sys.path.append("..")
from utils import load_corpus_and_vocab, create_corpus_and_vocab, create_co_matrix

def create_ppmi_matrix(co_matrix):
    """ Creates the positive pointwise mutual information (PPMI)
        matrix from a co-occurrence matrix.

        Returns a scipy.sparse.csr_matrix where matrix[i, j]
        is the PPMI of words with vocab index i and j.

        PPMI = max( Pr[i U j] / (Pr[i] * Pr[j]), 0 )
        Furthermore, we "smooth" the distribution by
        Pr[j] = (#j)^a / sum[(#w)^a].
        Levy and Goldberg 2015 suggest a = 0.75.
    """

    # Converts co-occurrence matrix to dok_matrix format for easier indexing.
    co_matrix = co_matrix.todok()

    # Sets size and creates matrix.
    size = co_matrix.shape[0]
    ppmi_matrix = dok_matrix((size, size))
    nonzero = co_matrix.nonzero()

    # Computes total and row sums with smoothing.
    co_total = co_matrix.sum()
    co_row_sum = co_matrix.sum(axis=1)
    co_row_sum_smoothed = [math.pow(count[0], 0.75) for count in co_row_sum]
    co_total_smoothed = np.sum(co_row_sum_smoothed)

    # Iterates over all nonzero entries in the co-occurrence matrix.
    for entry_index, (anchor_word_index, context_word_index) in enumerate(zip(*nonzero)):
        # Prints progress
        if entry_index % 1e5 == 0:
            print("Processing entry {:.2e}/{:.2e}".format(entry_index, len(nonzero[0])))

        # Gets total occurrences of each word.
        anchor_word_sum = co_row_sum.item(anchor_word_index)
        context_word_sum_smoothed = co_row_sum_smoothed.item(context_word_index)

        # Calculates probabilities.
        joint_prob = co_matrix[anchor_word_index, context_word_index] / co_total
        anchor_word_marginal_prob = anchor_word_sum / co_total
        context_word_marginal_prob = context_word_sum_smoothed / co_total_smoothed

        # Calculates PPMI.
        ratio = joint_prob / (anchor_word_marginal_prob * context_word_marginal_prob)
        pmi = math.log(ratio)
        ppmi = max(pmi, 0)

        # Updates PPMI values in the matrix.
        ppmi_matrix[anchor_word_index, context_word_index] = ppmi

    # Converts PPMI matrix to csr_matrix format.
    ppmi_matrix = ppmi_matrix.tocsr()

    # Creates directory if necessary.
    os.makedirs(os.path.dirname(FLAGS.ppmi_matrix_path), exist_ok=True)

    # Saves PPMI matrix.
    save_npz(FLAGS.ppmi_matrix_path, ppmi_matrix)
    print("Saved PPMI matrix.")

    return ppmi_matrix

def get_vocab_and_ppmi_matrix():
    """ Loads and/or creates the corpus and vocab,
        co-occurrence matrix, and the PPMI matrix.

        Returns the vocab and PPMI matrix.
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


    # Loads the PPMI matrix if possible.
    if os.path.isfile(FLAGS.ppmi_matrix_path):
        ppmi_matrix = load_npz(FLAGS.ppmi_matrix_path)
        print("Loaded vocab and PPMI matrix.\n")

    # Creates the PPMI matrix.
    else:
        print("Creating PPMI matrix...\n")
        ppmi_matrix = create_ppmi_matrix(co_matrix)
        print("Done creating PPMI matrix.\n")

    return vocab, ppmi_matrix

def n_dim_reduced_matrix(n_dim, matrix):
    """ Reduces dimensionality of the PPMI matrix via SVD. """

    # Instantiates an SVD and reduces the PPMI matrix.
    print("Creating SVD matrix...")
    svd = TruncatedSVD(n_components=n_dim, random_state=1)
    svd.fit(matrix)
    reduced_matrix = svd.transform(matrix)
    print("Done creating SVD matrix.\n")

    return svd, reduced_matrix

def main():
    _, ppmi_matrix = get_vocab_and_ppmi_matrix()
    _, weights = n_dim_reduced_matrix(100, ppmi_matrix)

    return weights

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

    parser.add_argument("--csv_path", default="/hdd0/datasets/fairness/articles.csv",
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
                              (default: ../saved/co_matrix.npz)")

    parser.add_argument("--ppmi_matrix_path", default="../saved/svd/ppmi_matrix.npz", type=str,
                        help="Path to PPMI matrix saved as a .npz file \
                              (default: ../saved/svd/ppmi_matrix.npz)")

    # Sets arguments as FLAGS
    FLAGS, _ = parser.parse_known_args()

    main()
