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
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk import PorterStemmer


train_set = pd.read_csv('train_set.csv', sep="\t", encoding = 'utf8')
test_set = pd.read_csv('test_set.csv', sep="\t", encoding = 'utf8')
train_set = train_set[0:100]

train_content = train_set['Content']

# Preprocess
# Stemming
sentences = []
sent = []
for sentence in train_content:
    for word in nltk.word_tokenize(sentence):
        sent.append(PorterStemmer().stem(word))
    sent = ' '.join(sent)
    sentences.append(sent)
    sent = []

# Label Encoding for the categories
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(train_set['Category'])

# Use a pipeline
# Transformer in scikit-learn - some class that have fit and transform method, or fit_transform method.
# Predictor - some class that has fit and predict methods, or fit_predict method.

pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS)),
    ('tfidf', TfidfTransformer()),
    ('svd', TruncatedSVD(n_components = 1000)),
    ('clf', RandomForestClassifier()),
])

predicted = pipeline.fit(sentences, y_train)

# Now evaluate all steps on test set and predict
predicted = pipeline.predict(test_set['Content'])

predicted_categories = le.inverse_transform(predicted)
