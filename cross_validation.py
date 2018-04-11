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
from sklearn.metrics import accuracy_score, precision_score
import nltk
from nltk import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import GaussianNB

# Main Program
# Classifier
clf =  RandomForestClassifier()

train_set = pd.read_csv('train_set.csv', sep="\t", encoding = 'utf8')
#train_set = train_set[0:500]
train_set_content = train_set['Content']
train_set_categories = train_set['Category']

# Accuracy mean value
mean_random_forest_accuracy = 0.0

kf = KFold(n_splits=10)
for train_indexes, test_indexes in kf.split(train_set):
    #print("TRAIN:", train_indexes, "TEST:", test_indexes)
    features_train = [train_set_content[i] for i in train_indexes]
    features_test = [train_set_content[i] for i in test_indexes]
    categories_train = [train_set_categories[i] for i in train_indexes]
    categories_test = [train_set_categories[i] for i in test_indexes]

    # Pipeline for features_train -> Content of every article
    vectorizer = TfidfVectorizer(stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)
    # Select only ten from it
    selector = SelectPercentile(f_classif, percentile = 10)
    selector.fit(features_train_transformed, categories_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    #print features_train_transformed
    features_test_transformed = selector.transform(features_test_transformed).toarray()
    #print features_test_transformed
    clf.fit(features_train_transformed, categories_train)
    prediction = clf.predict(features_test_transformed)
    acc = accuracy_score(prediction, categories_test)
    #print prediction
    #print categories_test
    precision = precision_score(prediction, categories_test, average=None)
    #print acc, precision
    mean_random_forest_accuracy += acc

mean_random_forest_accuracy /= 10

print "Random Forest Mean Accuracy: ", mean_random_forest_accuracy
