import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from cleaning_and_vectorization import X, tone, emotion, X_text_and_SR

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

keras = tf.keras



#HOLD-0UT DATA 
X_val, X_ho, y_val, y_ho = train_test_split(X, tone, stratify = tone, test_size = 0.15, random_state = 7)

#TRAIN TEST SPLIT BEFORE VECTORIZATION
X_train, X_test, y_train, y_test = train_test_split(X_val, y_val, stratify = y_val, test_size = 0.2, random_state = 7)

y_train = np.where(y_train == 'negative', -1, y_train)                                                                               
y_train = np.where(y_train == 'positive', 1, y_train) 
y_train = np.where(y_train == 'neutral', 0, y_train) 

y_test = np.where(y_test == 'negative', -1, y_test)                                                                               
y_test = np.where(y_test == 'positive', 1, y_test) 
y_test = np.where(y_test == 'neutral', 0, y_test)

#VECTORIZE TRAINING DATA
tfid_vectorizer = TfidfVectorizer()
tfid_vect = tfid_vectorizer.fit_transform(X_train).toarray()
words_tfid = tfid_vectorizer.get_feature_names()

#transforming X_test and X_holdout
tfid_test = tfid_vectorizer.transform(X_test).toarray()
tfid_ho = tfid_vectorizer.transform(X_ho).toarray()

#Build simple model

input_dim = tfid_vect.shape[1]
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)


def define_model(nb_filters, kernel_size, input_shape, pool_size):
    model = Sequential()  # model is a linear stack of layers (don't change)

    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     input_shape=input_shape))  # first conv. layer  KEEP
    model.add(Activation('relu'))  # Activation specification necessary for Conv2D and Dense layers

    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid'))  # 2nd conv. layer KEEP
    model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
    # model.add(Dropout(0.5))  # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Flatten())  # necessary to flatten before going into conventional dense layer  KEEP
    print('Model flattened out to ', model.output_shape)

    # now start a typical neural network
    model.add(Dense(32))  # (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))

    model.add(Dropout(0.5))  # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Dense(nb_classes))  # 10 final nodes (one for each class)  KEEP
    model.add(Activation('softmax'))  # softmax at end to pick between classes 0-9 KEEP

    # many optimizers available, see https://keras.io/optimizers/#usage-of-optimizers
    # suggest you KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    # and KEEP metrics at 'accuracy'
    # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # important inputs to the model: don't changes the ones marked KEEP
    batch_size = 50  # number of training samples used at a time to update the weights
    nb_classes = 3    # number of output possibilities: [0 - 9] KEEP
    nb_epoch = 10       # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 28, 28   # the size of the MNIST images KEEP
    input_shape = (tfid_vect.shape[0], tfid_vect.shape[1], 1)   # 1 channel image input (grayscale) KEEP
    nb_filters = 12    # number of convolutional filters to use
    pool_size = (2, 2)  # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (10, 10)  # convolutional kernel size, slides over image to learn features

    model = define_model(nb_filters, kernel_size, input_shape, pool_size)

    # during fit process watch train and test error simultaneously
    model.fit(tfid_vect, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(tfid_test, y_test))

    score = model.evaluate(tfid_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])  # this is the one we care about