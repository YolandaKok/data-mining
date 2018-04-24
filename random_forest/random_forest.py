import pandas as pd

import numpy as np

from sklearn import preprocessing
from sklearn.feature_extraction import text
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score

import nltk
from nltk import PorterStemmer

# Preprocess Data
# Stemming
def preprocess(content):
    # Preprocess
    # Stemming
    sentences = []
    sent = []
    for sentence in content:
        for word in nltk.word_tokenize(sentence):
            sent.append(PorterStemmer().stem(word))
        sent = ' '.join(sent)
        sentences.append(sent)
        sent = []
    return sentences

# Write results to csv
def write_to_csv(predictions):
    # Transform list of tuples to a dataframe
    df = pd.DataFrame(predictions, columns=['ID', 'Predicted_Category'])
    # Do not include index column
    df.to_csv("testSet_categories.csv", sep="\t", index=False)

# Main Program
train_set = pd.read_csv('../train_set.csv', sep="\t", encoding = 'utf8')
test_set = pd.read_csv('../test_set.csv', sep="\t", encoding = 'utf8')

train_content = train_set['Content'] + 2 * train_set['Title']
test_content = test_set['Content'] + 2 * test_set['Title']


# Preprocess data
sentences = preprocess(train_content)

# Label Encoding for the categories
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(train_set["Category"])

# Use a pipeline
# Transformer in scikit-learn - some class that have fit and transform method, or fit_transform method.
# Predictor - some class that has fit and predict methods, or fit_predict method.
pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS, max_features = 2000)),
    ('tfidf', TfidfTransformer()),
    ('svd', TruncatedSVD(n_components = 200)),
    ('clf', RandomForestClassifier()),
])

predicted = pipeline.fit(sentences, y_train)

# Now evaluate all steps on test set and predict
predicted = pipeline.predict(test_content)

predicted_categories = le.inverse_transform(predicted)
predictions = zip(test_set['Id'], predicted_categories)

# write to csv
write_to_csv(predictions)
