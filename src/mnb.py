
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from cleaning_and_vectorization import X, tone, emotion, X_text_and_SR
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

#HOLD-0UT DATA 
X_val, X_ho, y_val, y_ho = train_test_split(X, tone, stratify = tone, test_size = 0.15, random_state = 7)

#TRAIN TEST SPLIT BEFORE VECTORIZATION
X_train, X_test, y_train, y_test = train_test_split(X_val, y_val, stratify = y_val, test_size = 0.2, random_state = 7)

#VECTORIZE TRAINING DATA
tfid_vectorizer = TfidfVectorizer(max_features = 1000)
tfid_vect = tfid_vectorizer.fit_transform(X_train).toarray()
words_tfid = tfid_vectorizer.get_feature_names()

#transforming X_test and X_holdout
tfid_test = tfid_vectorizer.transform(X_test).toarray()
tfid_ho = tfid_vectorizer.transform(X_ho).toarray()

#NAIVE BAYES MODEL: Multinomial

mnb = MultinomialNB()
mnb.fit(tfid_vect, y_train)
mnb_yhat_train = mnb.predict(tfid_vect)
mnb_yhat_test = mnb.predict(tfid_test)

mnb_proba = mnb.predict_proba(tfid_test)

mnb_auc_score = roc_auc_score(y_test, mnb_proba, average = 'macro', multi_class='ovo')
mnb_class_rept = classification_report(y_test, mnb_yhat_test)
print(mnb_class_rept)
print(mnb_auc_score)
