import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer

# Read from csv file
train_set = pd.read_csv('train_set.csv', sep="\t", encoding = 'utf8')
#Preprocess Data
#print train_set[0:2]
train_content = train_set['Content']
print train_content.shape
sentences = []
sent = []
for sentence in train_content:
    for word in RegexpTokenizer(r"\b\w+\b").tokenize(sentence):
        sent.append(PorterStemmer().stem(word))
    sent = ' '.join(sent)
    sentences.append(sent)
    sent = []

#print len(sentences)
# Do classification S
clf = RandomForestClassifier()
# Pipeline the data



# Read the test_set
"""test_set = pd.read_csv('test_set.csv', sep="\t", encoding = 'utf8')
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(train_set["Category"])


clf = RandomForestClassifier(n_estimators=10)

clf.fit(X, y_train)
predicted = algorithm.predict(X)

# Vectorize the test_set
#count_vectorizer_test = CountVectorizer(stop_words = text.ENGLISH_STOP_WORDS)
# Maybe I have to also send the title
# How to add two dimensions
B = count_vectorizer_test.fit_transform(test_set['Content'])

le = preprocessing.LabelEncoder()
le.fit(train_set['Category'])
y = le.transform(train_set['Category'])

count_vectorizer = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS, max_features = 2000)
X = count_vectorizer.fit_transform(train_set['Content'])

# Classification
clf = RandomForestClassifier()

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(train_set["Category"])

#clf.fit(X, y)
# predict test set (here is the same as the train set)
clf.fit(B, y)
y_pred = clf.predict(B)

predicted_categories = le.inverse_transform(y_pred)
print classification_report(y, y_pred, target_names=list(le.classes_))
"""
