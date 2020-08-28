import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import xgboost.sklearn as xgb

from cleaning_and_vectorization import X, tone, emotion, X_text_and_SR

y_labels = LabelEncoder().fit_transform(tone)

#Train Test Split 
X_train, X_test, y_train, y_test  = train_test_split(X, y_labels, stratify = y_labels, test_size = 0.2, random_state = 7)

#VECTORIZE TRAINING DATA
tfid_vectorizer = TfidfVectorizer(max_features = 5000)
tfid_vect = tfid_vectorizer.fit_transform(X_train).toarray()
words_tfid = tfid_vectorizer.get_feature_names()

#transforming X_test and X_holdout
tfid_test = tfid_vectorizer.transform(X_test).toarray()


#XGBOOST CLASSIFIER

xg_final = xgb.XGBClassifier(objective = 'multi:softmax', num_class = 3, 
                            learning_rate = 0.1, n_estimators = 300,
                            min_child_weight = 3, max_depth = 6, gamma = 0.1, 
                            colsample_bynode = 0.75, subsample = 0.8)


xg_final.fit(tfid_vect, y_train)

xg_yhat_train = xg_final.predict(tfid_vect) 
xg_yhat_test = xg_final.predict(tfid_test)

xg_proba = xg_final.predict_proba(tfid_test)
xg_auc_score = roc_auc_score(y_test, xg_proba , average = 'weighted', multi_class = 'ovr')      
xg_class_rept = classification_report(y_test, xg_yhat_test)

print(xg_auc_score)
print(f'Test: {xg_class_rept}')
print(f'Train: {classification_report(y_train, xg_yhat_train)}')

#Create pandas dataframe and save predictions
xg_probabilities = pd.DataFrame(xg_proba)
xg_probabilities['predicted'] = xg_yhat_test
xg_probabilities['true'] = np.array(y_test)
xg_probabilities.columns = ['negative_prob', 'neutral_prob', 'positive_prob', 'predicted', 'true']

xg_probabilities.to_csv('../data/xg_probabilities.csv')


#Randomized GridSearch
'''

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
# max_depth = (3, 6)
# alphas = (0, 0.01)
# lambdas = (1, 5, 10)
# colsample_bynodes = (1, 0.75, 0.5)
# gammas = (0, 0.1, 0.3)
# min_child_weight = (1, 3)
# subsamples = (1, 0.8)
# n_estimators = (100, 300, 500)


gb_model = xgb.XGBClassifier(objective = 'multi:softmax', num_class = 3, 
                            learning_rate = 0.1, n_estimators = 300,
                            min_child_weight = 3, max_depth = 6, gamma = 0.1, 
                            colsample_bynode = 0.75, subsample = 0.8)

param_dict =dict(reg_lambda = lambdas)

# run randomized search
n_iter_search = 6
random_search = RandomizedSearchCV(gb_model, param_distributions=param_dict,
                                   n_iter=n_iter_search, n_jobs = -1, verbose = True, cv = 2)

start = time()
random_search.fit(tfid_vect, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

'''


#Tuning with GridSearchCV
'''
model = xgb.XGBClassifier(objective = 'multi:softmax', num_class = 3, learning_rate = 0.1, max_depth = 10, colsample_bynode=0.5, n_estimators = 100, reg_lambda = 10)
max_depth = (3, 6, 12)
# alphas = (0, 0.01)
lambdas = (1, 5, 10)
colsample_bynodes = (1, 0.8, 0.5)
gammas = (0, 0.01, 0.1, 1)


param_dict =dict(gamma = gammas)
kfold = StratifiedKFold(n_splits = 2, shuffle=True, random_state = 3)

def gs_tune(model, param_dict):
    grid_search = GridSearchCV(model, param_dict, scoring = 'neg_log_loss', n_jobs = -1, cv = kfold, verbose=3)
    result = grid_search.fit(tfid_vect, y_train)
    print(f'Best: {result.best_score_} using {result.best_params_}')

    means = result.cv_results_['mean_test_score']
    std = result.cv_results_['std_test_score']
    params = result.cv_results_['params']
    for m, s, p in zip(means, std, params):
        print(f'mean:{m}, stdev:{s}, param:{p}')
    return result.best_estimator_, result.best_params_

print(gs_tune(model, param_dict))
'''
