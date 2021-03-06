
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from cleaning_and_vectorization import X, tone, emotion, X_text_and_SR

#HOLD-0UT DATA 
X_train, X_test, y_train, y_test = train_test_split(X, tone, stratify = tone, test_size = 0.2, random_state = 7)

#VECTORIZE TRAINING DATA
tfid_vectorizer = TfidfVectorizer(max_features = 5000)
tfid_vect = tfid_vectorizer.fit_transform(X_train).toarray()
words_tfid = tfid_vectorizer.get_feature_names()

#transforming X_test and X_holdout
tfid_test = tfid_vectorizer.transform(X_test).toarray()

#NAIVE BAYES MODEL: Multinomial

mnb = MultinomialNB()
mnb.fit(tfid_vect, y_train)
mnb_yhat_train = mnb.predict(tfid_vect)
mnb_yhat_test = mnb.predict(tfid_test)

mnb_proba = mnb.predict_proba(tfid_test)

mnb_auc_score = roc_auc_score(y_test, mnb_proba, average = 'weighted', multi_class='ovr')
mnb_class_rept_test = classification_report(y_test, mnb_yhat_test)
mnb_class_rept_train = classification_report(y_train, mnb_yhat_train)

print(f'Test: {mnb_class_rept_test}')
print(f'Train: {mnb_class_rept_train}')
print(mnb_auc_score)

#Saving predictions to dataframe
mnb_probabilities = pd.DataFrame(mnb_proba)
mnb_probabilities['predicted'] = mnb_yhat_test
mnb_probabilities['true'] = np.array(y_test)
mnb_probabilities.columns = ['negative_prob', 'neutral_prob', 'positive_prob', 'predicted', 'true']
mnb_probabilities.to_csv('../data/mnb_probabilities.csv')
