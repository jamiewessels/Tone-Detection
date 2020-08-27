
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
from sklearn.decomposition import NMF

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


#RANDOM FOREST CLASSIFIER


rf_final = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 10, class_weight='balanced', max_features='sqrt', max_depth=15)
rf_final.fit(tfid_vect, y_train)


rf_yhat_train = rf_final.predict(tfid_vect) 
rf_yhat_test = rf_final.predict(tfid_test)
rf_proba = rf_final.predict_proba(tfid_test)  

rf_auc_score = roc_auc_score(y_test, rf_proba, average = 'macro', multi_class = 'ovo') 
rf_class_rept = classification_report(y_test, rf_yhat_test)
print(rf_class_rept)
print(rf_auc_score)

#create DataFrame of Probabilities
rf_probabilities = pd.DataFrame(rf_proba)
rf_probabilities['predicted'] = rf_yhat_test
rf_probabilities['true'] = np.array(y_test)
rf_probabilities.columns = ['negative_prob', 'neutral_prob', 'positive_prob', 'predicted', 'true']

rf_probabilities.to_csv('../data/rf_probabilities.csv')

#Tuning RF Using GSEARCH
'''
model = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 10, class_weight='balanced', max_features='sqrt', max_depth=15)

max_depths = (15, 5, None)
max_features = ('sqrt', 'auto')
n_estimators = (100, 300, 500)
max_leaf_nodes = (None, 40)
min_samples_leaf = (1, 5, 10)


param_dict =dict(min_samples_leaf = min_samples_leaf)
kfold = StratifiedKFold(n_splits = 2, shuffle=True, random_state = 3)

def rf_tune(model, param_dict):
    grid_search = GridSearchCV(model, param_dict, scoring = 'neg_log_loss', n_jobs = -1, cv = kfold, verbose=3)
    result = grid_search.fit(tfid_vect, y_train)
    print(f'Best: {result.best_score_} using {result.best_params_}')

    means = result.cv_results_['mean_test_score']
    std = result.cv_results_['std_test_score']
    params = result.cv_results_['params']
    for m, s, p in zip(means, std, params):
        print(f'mean:{m}, stdev:{s}, param:{p}')
    return result.best_estimator_, result.best_params_

print(rf_tune(model, param_dict))
'''

#Using NMF 

'''
nmf = NMF(n_components = 300, max_iter = 3000)
W = nmf.fit_transform(tfid_vect)
H = nmf.components_
recon_err = nmf.reconstruction_err_


nmf_train = W
nmf_test = nmf.transform(tfid_test)

rf_nmf = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 10, class_weight='balanced', max_features='sqrt', max_depth=15)
rf_nmf.fit(nmf_train, y_train)


rf_yhat_test = rf_nmf.predict(nmf_test)
rf_proba = rf_nmf.predict_proba(nmf_test)  

rf_auc_score = roc_auc_score(y_test, rf_proba, average = 'macro', multi_class = 'ovo') 
rf_class_rept = classification_report(y_test, rf_yhat_test)
print(rf_class_rept)
print(rf_auc_score)
'''

