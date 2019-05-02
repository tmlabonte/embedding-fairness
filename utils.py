import os
import pickle
from random import randrange
import re
import string

from nltk.corpus import stopwords
import numpy as np
from pandas import read_csv
from scipy.sparse import dok_matrix, save_npz

def article_preprocess(article):
    """ Preprocesses a raw article."""

    # Converts to lowercase.
    article = article.lower()

    # Removes punctuation.
    article = article.translate(str.maketrans("", "", string.punctuation))

    # Replaces pesky chars.
    article = article.replace('“', '').replace('”', '').replace('’', '\'')
    article = article.replace('£', '').replace('—', '').replace("\'", "")

    # Removes numbers.
    article = re.sub(r"\d+", "", article)

    # Removes single-char words.
    article = re.sub(r"\b[a-zA-Z]\b", "", article)

    # Removes excessive spaces.
    article = re.sub(" +", " ", article)

    return article

def corpus_preprocess(corpus, min_freq):
    """ Preprocesses corpus by removing rare words and stopwords"""

    # Creates a dictionary of word : count.
    counts = {}
    for article in corpus:
        for word in article.split():
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

    # Creates a dictionary of count : frequency.
    frequencies = {}
    for word in counts:
        if counts[word] in frequencies:
            frequencies[counts[word]] += 1
        else:
            frequencies[counts[word]] = 1

    # Creates set of rare words.
    rare_words = set()
    for word in list(counts.keys()):
        if counts[word] <= min_freq:
            rare_words.add(word)

    # Gets set of stopwords.
    stop_words = set(stopwords.words("english"))

    # Removes rare words and stopwords from each article.
    words_to_remove = rare_words.union(stop_words)
    for i, article in enumerate(corpus):
        result_words = [word for word in article.split() \
                        if word not in words_to_remove]
        corpus[i] = " ".join(result_words)

    return corpus

def create_corpus_and_vocab(csv_path, corpus_path, vocab_path,
                            num_articles, min_freq):
    """ Creates the corpus and vocab from the csv file.

        Extracts and preprocesses articles.
    """

    # Reads the csv file into a Pandas DataFrame.
    print("Loading csv...")
    df = read_csv(csv_path, engine="python",
                  encoding="utf-8", error_bad_lines=False)

    # Retrieves the first NUM_ARTICLES entries in the content column.
    corpus = list(df["content"])[:num_articles]
    print("Done loading csv.\n")

    # Preprocesses corpus.
    print("Preprocessing corpus...")
    corpus = [article_preprocess(article) for article in corpus]
    corpus = corpus_preprocess(corpus, min_freq)
    print("Done preprocessing corpus.\n")

    # Creates a vocab with ID's for O(1) lookup.
    vocab = {}
    for article in corpus:
        for word in article.split():
            vocab.setdefault(word, len(vocab))

    # Creates directories if necessary.
    os.makedirs(corpus_path, exist_ok=True)
    os.makedirs(vocab_path, exist_ok=True)

    # Saves the corpus and vocab.
    pickle.dump(corpus, open(corpus_path, "wb"))
    pickle.dump(vocab, open(vocab_path, "wb"))
    print("Saved corpus and vocab.")

    return corpus, vocab

def create_co_matrix(corpus, vocab, num_articles, co_matrix_path):
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
            print("Processing article {}/{}".format(article_index, num_articles))

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
    save_npz(co_matrix_path, matrix)
    print("Saved co-occurrence matrix.")

    return matrix

def load_corpus_and_vocab(corpus_path, vocab_path):
    """ Loads the corpus and vocab"""

    return pickle.load(open(corpus_path, "rb")), \
           pickle.load(open(vocab_path, "rb"))

def train_valid_test_split(data):
    """ Splits the dataset into training, validation, and testing
        sets with a 70/10/20 split.
    """

    # Splits the training set.
    splits = [int(.7 * len(data)), int(.8 * len(data)), len(data)]
    result = np.split(data, splits)

    # Extracts new datasets.
    train, valid, test = result[0], result[1], result[2]

    return train, valid, test
