
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from cleaning_and_vectorization import X, tone, emotion, X_text_and_SR
from sklearn.metrics import roc_auc_score, classification_report

from sklearn.neural_network import MLPClassifier

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, tone, stratify = tone, test_size = 0.2, random_state = 7)

# y_train = np.where(y_train == 'negative', -1, y_train)                                                                               
# y_train = np.where(y_train == 'positive', 1, y_train) 
# y_train = np.where(y_train == 'neutral', 0, y_train) 

# y_test = np.where(y_test == 'negative', -1, y_test)                                                                               
# y_test = np.where(y_test == 'positive', 1, y_test) 
# y_test = np.where(y_test == 'neutral', 0, y_test)

#VECTORIZE TRAINING DATA
tfid_vectorizer = TfidfVectorizer(max_features=5000)
tfid_vect = tfid_vectorizer.fit_transform(X_train).toarray()
words_tfid = tfid_vectorizer.get_feature_names()

#transforming X_test and X_holdout
tfid_test = tfid_vectorizer.transform(X_test).toarray()


#USING SKLEARN ML Perceptron
mlp = MLPClassifier(solver='adam', 
                    hidden_layer_sizes=(32, 8), random_state=1, alpha = .5,
                    max_iter = 300, verbose = True)

mlp.fit(tfid_vect, y_train)


mlp_pred_test = mlp.predict(tfid_test)
mlp_pred_train = mlp.predict(tfid_vect)


print(roc_auc_score(y_test, mlp.predict_proba(tfid_test), average = 'weighted', multi_class = 'ovr'))
print(f'Train: {classification_report(y_train, mlp_pred_train)}')
print(f'Test: {classification_report(y_test, mlp_pred_test)}')