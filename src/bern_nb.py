
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from cleaning_and_vectorization import X, tone, emotion, X_text_and_SR

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, tone, stratify = tone, test_size = 0.2, random_state = 7)

#VECTORIZE TRAINING DATA
tfid_vectorizer = TfidfVectorizer(max_features = 5000)
tfid_vect = tfid_vectorizer.fit_transform(X_train).toarray()
words_tfid = tfid_vectorizer.get_feature_names()

#transforming X_test and X_holdout
tfid_test = tfid_vectorizer.transform(X_test).toarray()


#NAIVE BAYES MODEL: Bernoulli

bnb = BernoulliNB()
bnb.fit(tfid_vect, y_train)
bnb_yhat_train = bnb.predict(tfid_vect)
bnb_yhat_test = bnb.predict(tfid_test)

bnb_proba = bnb.predict_proba(tfid_test)

bnb_auc_score = roc_auc_score(y_test, bnb_proba, average = 'weighted', multi_class = 'ovr')
bnb_class_rept_test = classification_report(y_test, bnb_yhat_test)
bnb_class_rept_train = classification_report(y_train, bnb_yhat_train)

print(f'Test: {bnb_class_rept_test}')
print(f'Train: {bnb_class_rept_train}')
print(bnb_auc_score)

#Saving predictions to dataframe
bnb_probabilities = pd.DataFrame(bnb_proba)
bnb_probabilities['predicted'] = bnb_yhat_test
bnb_probabilities['true'] = np.array(y_test)
bnb_probabilities.columns = ['negative_prob', 'neutral_prob', 'positive_prob', 'predicted', 'true']
bnb_probabilities.to_csv('../data/bnb_probabilities.csv')
