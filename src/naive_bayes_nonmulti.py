import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB, BernoulliNB, CategoricalNB
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

#NAIVE BAYES MODEL: Multinomial
gnb = GaussianNB()
cnb = ComplementNB()
bnb = BernoulliNB()
catnb = CategoricalNB()

def test_naive_bayes(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    proba = model.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, proba, average = 'weighted', multi_class='ovr') 
    class_rept = classification_report(y_test, y_hat_test)
    print(f'{str(model)} Report: {class_rept}')
    print(f'{str(model)} AUC: {auc_score}')
    return class_rept

print(f'Gaussian........:')
print(test_naive_bayes(gnb, tfid_vect, y_train, tfid_test, y_test))

print(f'Binomial........:')
print(test_naive_bayes(bnb, tfid_vect, y_train, tfid_test, y_test))

print(f'Complement........:')
print(test_naive_bayes(cnb, tfid_vect, y_train, tfid_test, y_test))

print(f'Categorical........:')
print(test_naive_bayes(catnb, tfid_vect, y_train, tfid_test, y_test))


#Code for putting predictions in dataframe
'''
xx_probabilities = pd.DataFrame(xx)
xx_probabilities['predicted'] = xx_yhat_test
xx_probabilities['true'] = np.array(y_test)
xx_probabilities.columns = ['negative_prob', 'neutral_prob', 'positive_prob', 'predicted', 'true']
xx_probabilities.to_csv('../data/xx_probabilities.csv')
'''
