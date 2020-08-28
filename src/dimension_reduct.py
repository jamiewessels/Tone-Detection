import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split
from cleaning_and_vectorization import X, tone, emotion, X_text_and_SR
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

y_labels = LabelEncoder().fit_transform(tone)



#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, tone, stratify = tone, test_size = 0.2, random_state = 7)


#VECTORIZE TRAINING DATA
tfid_vectorizer = TfidfVectorizer(max_features = 5000)
tfid_vect = tfid_vectorizer.fit_transform(X_train).toarray()
words_tfid = tfid_vectorizer.get_feature_names()


#transforming X_test and X_holdout
tfid_test = tfid_vectorizer.transform(X_test).toarray()

#Save numpy zip file
'''
np.savez('tfid_vect', tfid_vect)
np.savez('tfid_test', tfid_test)
np.savez('y_test', y_test)
np.savez('y_train', y_train)
'''


#most influential words (tfid matrix)
'''
top_idx = np.argsort(np.sum(tfid_vect, axis = 0))[::-1][:20]
top_words = np.array(words_tfid)[top_idx]
'''

#####EXPLORE DIMENSION REDUCTION#####
#PCA 100 Components
'''
pca = PCA(n_components = 100)
pca.fit_transform(tfid_vect)
exp_var_ratio = pca.explained_variance_ratio_
exp_sing_vals = pca.singular_values_
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


#NMF EXPLORATION
nmf = NMF(n_components = 30, max_iter = 3000)
W = nmf.fit_transform(tfid_vect)
H = nmf.components_
recon_err = nmf.reconstruction_err_

# print(recon_err)

topic_labels = []
for i, row in enumerate(H):
    top_ten = list(np.argsort(row)[::-1][:10])
    # print(top_ten)
    # print('topic', i)
    print(f'Latent Topic #{i+1}: {np.array(words_tfid)[top_ten]}')
    topic_labels.append(top_ten)
nmf_train = W
nmf_test = nmf.transform(tfid_test)