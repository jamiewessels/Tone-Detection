import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from cleaning_and_vectorization import X, tone, emotion, X_text_and_SR
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier


y_labels = LabelEncoder().fit_transform(tone)

#HOLD-0UT DATA 
X_val, X_ho, y_val, y_ho = train_test_split(X, y_labels, stratify = y_labels, test_size = 0.15, random_state = 7)

#TRAIN TEST SPLIT BEFORE VECTORIZATION
X_train, X_test, y_train, y_test = train_test_split(X_val, y_val, stratify = y_val, test_size = 0.2, random_state = 7)

#VECTORIZE TRAINING DATA
tfid_vectorizer = TfidfVectorizer(max_features = 1000)
tfid_vect = tfid_vectorizer.fit_transform(X_train).toarray()
words_tfid = tfid_vectorizer.get_feature_names()

#transforming X_test and X_holdout
tfid_test = tfid_vectorizer.transform(X_test).toarray()
tfid_ho = tfid_vectorizer.transform(X_ho).toarray()

#DataMatrix for XGBoost
# data_dmatrix = xgb.DMatrix(data=tfid_vect,label=y_train)



#XGBOOST CLASSIFIER

xg_final = xgb.XGBClassifier(objective = 'multi:softmax', num_class = 3, 
                        learning_rate=0.1, n_estimators = 200, verbosity = 3, 
                        max_depth = 6, colsample_bynode=0.5)

xg_final.fit(tfid_vect, y_train)

xg_yhat_train = xg_final.predict(tfid_vect) 
xg_yhat_test = xg_final.predict(tfid_test)

xg_proba = xg_final.predict_proba(tfid_test)
xg_auc_score = roc_auc_score(y_test, xg_proba , average = 'macro', multi_class = 'ovo')      
xg_class_rept = classification_report(y_test, xg_yhat_test)
print(xg_class_rept)
print(xg_auc_score)


# {'true_label': y_test, 'predicted_label': xg_yhat_test}


#TUNING MODEL USING G-SEARCH
'''
model = xgb.XGBClassifier(objective = 'multi:softmax', num_class = 3, learning_rate=0.1, max_depth = 6, colsample_bynode=0.5)
max_depth = (3, 6, 9)
n_estimators = (100, 200)
alphas = (0, 0.01)
colsample_bynodes = (1, 0.8, 0.5)

param_dict =dict(n_estimators = n_estimators)
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
