import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from cleaning_and_vectorization import X, tone, emotion, X_text_and_SR
from sklearn.metrics import roc_auc_score, classification_report

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from tensorflow.keras.utils import to_categorical
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
tfid_vectorizer = TfidfVectorizer(max_features=5000)
tfid_vect = tfid_vectorizer.fit_transform(X_train).toarray()
words_tfid = tfid_vectorizer.get_feature_names()

#transforming X_test and X_holdout
tfid_test = tfid_vectorizer.transform(X_test).toarray()
tfid_ho = tfid_vectorizer.transform(X_ho).toarray()

#Build simple model
model2 = Sequential()

input_dim = tfid_vect.shape[1]
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

# model1.add(SimpleRNN(32,  input_shape = reshaped_x.shape[1:], return_sequences=True, activation = 'relu'))

# model1.add(SimpleRNN(32,  return_sequences=False, activation = 'relu'))

model2.add(Dense(128, input_dim = input_dim, activation = 'relu'))

model2.add(Dense(64, activation = 'relu'))

model2.add(Dense(32, activation = 'relu'))

model2.add(Dense(32, activation = 'relu'))

model2.add(Dense(16, activation = 'relu'))

model2.add(Dense(3, activation = 'softmax'))

model2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model2.fit(tfid_vect, y_train, epochs = 1000, batch_size = 500, verbose = 10, validation_data=(tfid_test, y_test), validation_split=0.2)

acc_m2 = model2.evaluate(tfid_test, y_test, verbose = 10)

# m1_pred_proba_train = model1.predict(tfid_vect)
m2_pred_proba_test = model2.predict(tfid_test)
m2_pred_test = model2.predict_classes(tfid_test)
true = np.argmax(y_test, axis = 1) 
predictions = np.argmax(m2_pred_proba_test, axis = 1) 
print(model2.summary())
print(roc_auc_score(y_test, m2_pred_proba_test))
print(classification_report(true, predictions))