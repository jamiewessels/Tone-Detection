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

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_treebank_pos_tagger')

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()

def remove_stopwords_punc_numbers(stopwords, punc_lst, input_lst):
    no_stop = [[word for word in words if word not in stopwords]for words in input_lst]
    no_punc = [[word for word in words if word not in punc_lst]for words in no_stop]
    no_num = [[word for word in words if not word.isnumeric()] for words in no_punc]
    final = [[word for word in words if word.isalnum()] for words in no_num]
    return final

#LOAD DF
df = pd.read_csv('../data/grouped_emotions.csv')
#dropping columns to prevent data leakage (remove user info and rater feelings about text)
df = df.drop(columns = ['Unnamed: 0', 'author', 'id', 'link_id', 'parent_id', 'example_very_unclear'])

#OUTPUTS
tone = df.pop('tone')
emotion = df.pop('emotion_classification')

#INPUTs
df_cols = df.columns
#subreddits and text
X_text_and_SR = df.values
X = df['text'].values

#TOKENIZE AND CLEAN
stop = set(stopwords.words('english'))
punc_lst = list(string.punctuation)
punc_lst.extend(['`', "''", '``', '...', '....', '.....'])

X = [remove_accents(x) for x in X]
X = [word_tokenize(i.lower()) for i in X]
X = remove_stopwords_punc_numbers(stop, punc_lst, X)

#LEMMATIZE
wordnet = WordNetLemmatizer()
X = [[wordnet.lemmatize(word) for word in words] for words in X]

#STEMMATIZE
# stemmer_porter = PorterStemmer()
# X = [[stemmer_porter.stem(word) for word in words] for words in X]

for element in X: 
    ' '.join(element)
X = [" ".join(x) for x in X]







