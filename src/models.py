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
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from cleaning_and_vectorization import X, tone, emotion, X_text_and_SR


#TRAIN TEST SPLIT BEFORE VECTORIZATION
X_train, X_test, y_train, y_test = train_test_split(X, tone, stratify = tone, test_size = 0.2)

#VECTORIZE TRAINING DATA
tfid_vectorizer = TfidfVectorizer()
tfid_vect = tfid_vectorizer.fit_transform(X_train).toarray()
words_tfid = tfid_vectorizer.get_feature_names()

count_vectorizer = CountVectorizer()
count_vect = count_vectorizer.fit_transform(X_train).toarray()
words_count = count_vectorizer.get_feature_names()

#QUICK ANALYSIS: largest word counts (top 20)
#TODO WORD CLOUD? 
top_idx = np.argsort(np.sum(count_vect, axis = 0))[::-1][:20]
top_words = np.array(words_count)[top_idx]