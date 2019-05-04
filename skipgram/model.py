from keras.layers import Input, Embedding, Reshape, Dense, dot
from keras.models import Model

def create_model(vector_dim, vocab_size):
    """ Creates a Keras model which runs an anchor word and a context word
        through an embedding layer to determine if the context word
        is real or negatively sampled.

        Credit to https://github.com/adventuresinML/
        adventures-in-ml-code/blob/master/keras_word2vec.py
    """

    # Creates input variables.
    anchor_index = Input((1,), name="anchor_index")
    context_index = Input((1,), name="context_index")

    # Creates embedding layer.
    embedding = Embedding(vocab_size, vector_dim,
                          input_length=1, name="embedding")

    # Gets embeddings of input words.
    anchor_vector = embedding(anchor_index)
    anchor_vector = Reshape((vector_dim, 1))(anchor_vector)
    context_vector = embedding(context_index)
    context_vector = Reshape((vector_dim, 1))(context_vector)

    # Takes the dot product of the anchor and context
    # as a similarity measure.
    dot_product = dot([anchor_vector, context_vector], axes=1)
    dot_product = Reshape((1,))(dot_product)

    # Creates sigmoid output node.
    # Closer to 1 means higher probability of
    # the context word being real.
    prob_of_real = Dense(1, activation="sigmoid")(dot_product)

    # Compiles the model.
    model = Model(input=[anchor_index, context_index], output=prob_of_real)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model
