import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk 
import string
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from cleaning_and_vectorization import X, tone, emotion, X_text_and_SR
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, classification_report


#TRAIN TEST SPLIT BEFORE VECTORIZATION
X_train, X_test, y_train, y_test = train_test_split(X, tone, stratify = tone, test_size = 0.2)

#VECTORIZE TRAINING DATA
tfid_vectorizer = TfidfVectorizer()
tfid_vect = tfid_vectorizer.fit_transform(X_train).toarray()
words_tfid = tfid_vectorizer.get_feature_names()
tfid_test = tfid_vectorizer.transform(X_test).toarray()

count_vectorizer = CountVectorizer()
count_vect = count_vectorizer.fit_transform(X_train).toarray()
words_count = count_vectorizer.get_feature_names()

#QUICK ANALYSIS: largest word counts (top 20)
#TODO WORD CLOUD? 
'''
top_idx = np.argsort(np.sum(count_vect, axis = 0))[::-1][:20]
top_words = np.array(words_count)[top_idx]
'''


#####EXPLORE DIMENSION REDUCTION#####
#PCA 100 Components
'''
pca = PCA(n_components = 100)
pca.fit_transform(tfid_vect)
exp_var_ratio = pca.explained_variance_ratio_
exp_sing_vals = pca.singular_values_
'''

#PCA 5,000 Components
'''
pca = PCA(n_components = 5000)
pca.fit_transform(tfid_vect)
exp_var_ratio = pca.explained_variance_ratio_
'''

#PCA Plots: explained variance ratios
'''
xs = np.arange(1, 21) 
ratios = (exp_var_ratio[:20]).round(3)  

fig, ax = plt.subplots(figsize = (8,8))
ax.plot(xs, ratios)
ax.set_xticks(xs)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Proportion of Variance Explained')
ax.set_title('Explained Variance by Principal Components')
ax.set_ylim(0, 1.0)
fig.savefig('../images/PCA_scree.jpeg')
fig.show()

cum_var_ratio = np.cumsum(exp_var_ratio)
xs = np.arange(1, 101)
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(xs, cum_var_ratio)
ax.set_xticks(np.arange(0, 101, 10))
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Proportion of Variance Explained')
ax.set_title('Cumulative Explained Variance by Number of Principal Components')
ax.set_ylim(0, 1.0)
fig.savefig('../images/PCA_cum_scree.jpeg')
fig.show()
'''


#SVD
'''
U, sigma, V = np.linalg.svd(tfid_vect)
sigma_sq = sigma ** 2

lf = 99
s = sigma[:lf]
s_sq = s ** 2
u = U[:, :lf]
v = V[:lf, :] 

sigma_cum = np.cumsum(s_sq)*100/ np.sum(sigma_sq)
'''


#NAIVE BAYES MODEL: Multinomial
'''
mnb = MultinomialNB()
mnb.fit(tfid_vect, y_train)
mnb_yhat_train = mnb.predict(tfid_vect)
mnb_yhat_test = mnb.predict(tfid_test)
mnb_acc_train = mnb.score(tfid_vect, y_train)
mnb_acc_test = mnb.score(tfid_test, y_test)
mnb_f1_macro_train = f1_score(y_train, mnb_yhat_train, average = 'macro') 
mnb_f1_macro_test = f1_score(y_test, mnb_yhat_test, average = 'macro') 

mnb_proba = mnb.predict_proba(tfid_test)
mnb_auc_score = roc_auc_score(y_test, mnb_proba, average = 'macro', multi_class='ovo')
'''

#GRADIENT BOOSTING CLASSIFIER
'''
gb=GradientBoostingClassifier(n_estimators=200,learning_rate=0.1)
gb.fit(tfid_vect,y_train)
'''

#RANDOM FOREST CLASSIFIER

rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1, verbose = 10)
rf.fit(tfid_vect, y_train)

rf_acc_train = rf.score(tfid_vect, y_train) 
rf_acc_test = rf.score(tfid_test, y_test)
rf_yhat_train = rf.predict(tfid_vect) 
rf_yhat_test = rf.predict(tfid_test)
rf_f1_macro_train = f1_score(y_train, rf_yhat_train, average = 'macro')  
rf_f1_macro_test = f1_score(y_test, rf_yhat_test, average = 'macro')    

rf_proba = rf.predict_proba(tfid_test)  
rf_auc_score = roc_auc_score(y_test, rf_proba, average = 'macro', multi_class = 'ovo') 


#DECISION TREE
'''
max_depths = [5, 10, 20, 50]
for depth in max_depths:
    print('next tree: \n')
    tree = DecisionTreeClassifier(max_depth = depth)
    tree.fit(tfid_vect, y_train)
    print(f'{depth}: train: {tree.score(tfid_vect, y_train)} test: {tree.score(tfid_test, y_test)}')
'''



'''
cross_val_score(estimator=gb,X=tfid_test,y=y_test,scoring='f1_weighted',cv=5)
class_rept = classification_report(y_true=y_test,y_pred=gb.predict(tfid_test))

mnb_yhat_train = mnb.predict(tfid_vect)
mnb_yhat_test = mnb.predict(tfid_test)
mnb_acc_train = mnb.score(tfid_vect, y_train)
mnb_acc_test = mnb.score(tfid_test, y_test)
mnb_f1_macro_train = f1_score(y_train, mnb_yhat_train, average = 'macro') 
mnb_f1_macro_test = f1_score(y_test, mnb_yhat_test, average = 'macro') 

mnb_proba = mnb.predict_proba(tfid_test)
mnb_auc_score = roc_auc_score(y_test, mnb_proba, average = 'macro', multi_class='ovo')
'''