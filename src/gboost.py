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
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier



#HOLD-0UT DATA 
X_val, X_ho, y_val, y_ho = train_test_split(X, tone, stratify = tone, test_size = 0.15, random_state = 7)

#TRAIN TEST SPLIT BEFORE VECTORIZATION
X_train, X_test, y_train, y_test = train_test_split(X_val, y_val, stratify = y_val, test_size = 0.2, random_state = 7)

#VECTORIZE TRAINING DATA
tfid_vectorizer = TfidfVectorizer(max_features = 5000)
tfid_vect = tfid_vectorizer.fit_transform(X_train).toarray()
words_tfid = tfid_vectorizer.get_feature_names()

#transforming X_test and X_holdout
tfid_test = tfid_vectorizer.transform(X_test).toarray()
tfid_ho = tfid_vectorizer.transform(X_ho).toarray()

count_vectorizer = CountVectorizer()
count_vect = count_vectorizer.fit_transform(X_train).toarray()
words_count = count_vectorizer.get_feature_names()

#DataMatrix for XGBoost
# data_dmatrix = xgb.DMatrix(data=tfid_vect,label=y_train)

#XGBOOST CLASSIFIER
'''
xg = xgb.XGBClassifier(objective = 'multi:softmax', num_class = 3, 
                        learning_rate=0.1, n_estimators = 200, verbosity = 3, subsample=1,
                        eval_metric = 'merror')
#look into max depth, alpha (lasso), lambda (l2) look at others
xg.fit(tfid_vect, y_train)

xg_yhat_train = xg.predict(tfid_vect) 
xg_yhat_test = xg.predict(tfid_test)

xg_proba = xg.predict_proba(tfid_test)
xg_auc_score = roc_auc_score(y_test, xg_proba , average = 'macro', multi_class = 'ovo')      
xg_class_rept = classification_report(y_test, xg_yhat_test)
print(xg_class_rept)
print(xg_auc_score)
'''

# {'true_label': y_test, 'predicted_label': xg_yhat_test}

#Grid Search CV

clf = xgb.XGBClassifier(objective = 'multi:softmax', num_class = 3)

clf_parms = {'learning_rate':[0.1], 'n_estimators':[300], 'max_depth':[6, 20], 'alpha': [0, 0.01], 'colsample_bynode':[0.5, 1], 'eval_metric': ['merror', 'mlogloss']}
xb_gsearch = GridSearchCV(clf, clf_parms, scoring ='neg_log_loss', n_jobs=-1)
xb_gsearch.fit(tfid_vect, y_train)
best_xb = xb_gsearch.best_params_
print(f'XB best estimator: {xb_gsearch.best_estimator_}')