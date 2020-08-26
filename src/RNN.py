import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from cleaning_and_vectorization import X, tone, emotion, X_text_and_SR

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, RNN, Conv1D
from tensorflow.keras.utils import to_categorical
import tensorflow.data
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

train_dataset = tf.data.Dataset.from_tensor_slices((tfid_vect, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((tfid_test, y_test))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

#TODO: PADDING!!!!!!!! THAT'S WHY THIS AINT WORKINNNNN

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(10, 50,activation = 'relu', input_shape=(tfid_vect.shape[0], input_dim, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3)
])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy,
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


model.fit(tfid_vect, y_train, epochs=10,
                    validation_steps=30, verbose = 10, 
                    validation_data=(tfid_test, y_test))

# model.summary()

'''
model = Sequential()

# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=input_dim, output_dim=64))

#The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(layers.SimpleRNN(128, return_sequences=True))

model.add(Dense(32, input_dim = input_dim, activation = 'relu'))

model.add(Dense(3, activation = 'softmax'))

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"])
'''



# model1.add(SimpleRNN(32,  input_shape = reshaped_x.shape[1:], return_sequences=True, activation = 'relu'))

# model1.add(SimpleRNN(32,  return_sequences=False, activation = 'relu'))

# model2.add(Dense(500, input_dim = input_dim, activation = 'relu'))

# model2.add(Dense(32, activation = 'relu'))

# model2.add(Dense(32, activation = 'relu'))

# model2.add(Dense(3, activation = 'softmax'))

# model2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# model.fit(tfid_vect, y_train, epochs = 10, batch_size = 10, verbose = 10, validation_data=(tfid_test, y_test), validation_split=0.2)

# acc_m1 = model.evaluate(tfid_test, y_test, verbose = 10)

# m1_pred_proba_train = model1.predict(tfid_vect)
# m1_pred_proba_test = model1.predict(tfid_test)
# m1_pred_test = model1.predict_classes(tfid_test)
# print(model2.summary())