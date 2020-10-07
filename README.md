# A Study in Embedding Fairness
Fairness is a massive problem when working with word embeddings. For example, non-gendered words like "programmer" and "homemaker" are heavily associated with certain genders, as in the landmark 2016 paper ["Man is to Computer Programmer as Woman is to Homemaker"](https://arxiv.org/abs/1607.06520), which heavily inspired this study (joint with CAIS++ at the University of Southern California).

### Part 1: Data preprocessing
We're interested in how biases appear in real life media, so we analyzed [a dataset of 50,000 news articles from Kaggle](https://www.kaggle.com/snapcrack/all-the-news#articles3.csv). The downside is that these articles are quite raw and need a lot of preprocessing, so we spent a lot of time carefully cleaning and shaping the dataset. Then, we had to develop a vocabulary and build a secondary dataset of anchor words and their contextual information.

### Part 2: Comparing methods of creating word embeddings
Inspired by [Levy and Goldberg 2015](https://www.aclweb.org/anthology/Q15-1016), we built both count-based (PPMI/SVD) methods and neural network-based (skipgram) methods, then compared the results.

#### PPMI/SVD
The PPMI technique involves building a massive sparse matrix of word co-occurrences, then analyzing the entries in a Bayesian manner. From there, we used singular value decomposition (SVD) to reduce the dimensionality to 1% of the original, thereby projecting the words onto a vector space. This model generally outperformed skipgram in terms of rare word representation and semantic relationships, but took up a lot of memory and we had to use clever data structures to hold all the information.

#### Skipgram
The skipgram technique involves a neural network which attempts to predict context words given an anchor word. After training a feedforward network with one layer, we extracted the weights matrix and matched them with their corresponding words. This model wasn't quite as good as PPMI/SVD because it solves an approximate PPMI problem as it approaches convergence, but it was slow to converge. However, as the dataset size increases, skipgram becomes much more efficient than PPMI.

### Part 3: Bias analysis results
We analyzed bias using the Euclidean/cosine distance between word vectors. A sensitive word is considered unbiased if it is equidistant from the sensitive classes in the vector space. As my results showed, this dataset is heavily biased, with extremely negative non-gendered words close to both "men" and "women". Here is a selection of the top closest words to "men" and "women" using the SVD model with 80 dimensions:

men: male, young, boy, girl, **alone**, youth, teens, **brutally**, taken, lives, **robbed**, wives, witnessed

women: **bullied**, male, female, **bisexual**, **trans**, **genital**, young, **lesbian**, **sex**, **disabilities**, **unwanted**, **abused**

### Part 4: Debiasing
During our initial attempt at debiasing, we studied hard and soft debiasing as mentioned in the first paper. This involves creating an optimization problem over the weights matrix where we want to balance centering sensitive words within sensitive classes (e.g., non-gendered words within gender classes) and preserving the original word embeddings (e.g., we want the word "president" to be equidistant from "man" and "woman", but we also want "president" to remain close to "vice-president"). We're also interested in how we can use regularization to enforce a fairness constraint during the training process.
