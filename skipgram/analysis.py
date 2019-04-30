import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import pickle
import operator

model = Sequential([
    Dense(32, input_shape=(11231,)),
    Dense(11231),
    Activation('softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.load_weights('weights.h5')

weights = model.layers[1].get_weights()
vectors = weights[0] + weights[1]
vectors = vectors.transpose()

vocab = pickle.load(open('vocab.pickle', 'rb'))

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def distance_between(word1_index, word2_index, context_word_index, vectors):
    return (euclidean_dist(vectors[context_word_index], vectors[word1_index]),
            euclidean_dist(vectors[context_word_index], vectors[word2_index]))

def n_closest(n, word_index, vocab, vectors):
    closest = {}
    for index, vector in enumerate(vectors):
        if index != word_index:
            dist = euclidean_dist(vectors[index], vectors[word_index])

            if len(closest) > 0:
                max_key_in_closest = max(closest, key=closest.get)
                max_val_in_closest = closest[max_key_in_closest]

            if len(closest) < n:
                closest[index] = dist
            elif dist < max_val_in_closest:
                del closest[max_key_in_closest]
                closest[index] = dist

    for index, dist in closest.items():
        for word, vocab_index in vocab.items():
            if index == vocab_index:
                closest[word] = dist
                del closest[index]
                break

    sorted_closest = sorted(closest.items(), key=operator.itemgetter(1))
    sorted_closest = [pair[0] for pair in sorted_closest]

    return sorted_closest

def n_biggest_difference(n, word1index, word2index, vocab, vectors):
    diffs = []
    for word in list(vocab.keys()):
        if vocab[word] is not word1index and vocab[word] is not word2index:
            dist = distance_between(word1index, word2index, vocab[word], vectors)
            diff = (word, dist[0] - dist[1])
            diffs.append(diff)

    sorted_diffs = sorted(diffs, key=lambda tup:abs(tup[1]), reverse=True)

    return sorted_diffs[:n]

#print(distance_between(vocab["man"], vocab["woman"], vocab["president"], vectors))
#print(distance_between
print(n_closest(300, vocab["men"], vocab, vectors))
#print()
print(n_closest(300, vocab["women"], vocab, vectors))
#print()
#print(n_biggest_difference(50, vocab["man"], vocab["woman"], vocab, vectors))
