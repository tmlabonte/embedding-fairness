import re
import string
import os
import pickle
from pandas import read_csv
from nltk.corpus import stopwords

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

def create_corpus_and_vocab(data_path, corpus_path, vocab_path,
                            num_articles, min_freq):
    """ Creates the corpus and vocab from the dataset.

        Extracts and preprocesses articles.
    """

    # Reads the dataset into a Pandas DataFrame.
    print("Loading data...")
    df = read_csv(data_path, engine="python",
                  encoding="utf-8", error_bad_lines=False)

    # Retrieves the first NUM_ARTICLES entries in the content column.
    corpus = list(df["content"])[:num_articles]
    print("Done loading data.\n")

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

def load_corpus_and_vocab(corpus_path, vocab_path):
    """ Loads the corpus and vocab"""

    return pickle.load(open(corpus_path, "rb")), \
           pickle.load(open(vocab_path, "rb"))
