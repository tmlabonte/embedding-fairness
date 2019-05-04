import argparse
import pickle

import numpy as np
from scipy.sparse import load_npz

from svd.main import n_dim_reduced_matrix

def dist(word_vec1, word_vec2):
    """ Returns the distance between two word vectors. """

    # Calculates vector distance via the norm of the difference.
    return np.linalg.norm(word_vec1 - word_vec2)

def n_closest_words(word, n, weights, vocab, reversed_vocab):
    """ Returns the n closest words to a given target word. """

    # Gets the word vector for the target word.
    word_row = weights[vocab[word]]

    # Calculates the distance between the target word and all other words.
    dists = np.asarray([dist(word_row, row) for row in weights])

    # Selects the n indices with minimal distance.
    top_n_indices = np.argpartition(dists, n + 1)[:n + 1]

    # Transforms the indices into their corresponding words.
    top_n_words = [reversed_vocab[index] for index in top_n_indices if index != vocab[word]]

    return top_n_words

def main():
    vocab = pickle.load(open(FLAGS.vocab_path, "rb"))
    reversed_vocab = dict(zip(vocab.values(), vocab.keys()))

    if FLAGS.alg == "svd":
        ppmi_matrix = load_npz(FLAGS.ppmi_matrix_path)
        _, weights = n_dim_reduced_matrix(100, ppmi_matrix)
    elif FLAGS.alg == "skipgram":
        weights = pickle.load(FLAGS.weights_path)
        print(weights.shape)
    else:
        raise ValueError("Invalid argument for alg (choose svd or skipgram).")

    top30_words_men = n_closest_words("men", 30, weights, vocab, reversed_vocab)
    top30_words_women = n_closest_words("women", 30, weights, vocab, reversed_vocab)

    print("Top 30 Words Associated with MEN: " + str(top30_words_men))
    print("Top 30 Words Associated with WOMEN: " + str(top30_words_women))

if __name__ == "__main__":
    # Instantiates an argument parser
    parser = argparse.ArgumentParser()

    # Adds command line arguments
    parser.add_argument("--alg", default="svd", type="str",
                        help="Which algorithm to analyze: either svd or \
                              skipgram (default: svd)")

    parser.add_argument("--vocab_path", default="../saved/vocab.pickle", type=str,
                        help="Path to vocab saved as a .pickle file \
                              (default: ../saved/vocab.pickle)")

    parser.add_argument("--ppmi_matrix_path", default="../saved/svd/ppmi_matrix.npz", type=str,
                        help="Path to PPMI matrix saved as a .npz file \
                              (default: ../saved/svd/ppmi_matrix.npz)")

    parser.add_argument("--weights_path",
                        default="../saved/skipgram/weights/weights-10.hdf5",
                        type=str, help="Path to network weights folder \
                                        (default: ../saved/skipgram/weights/\
                                        weights-10.hdf5)")

    # Sets arguments as FLAGS
    FLAGS, _ = parser.parse_known_args()

    main()
