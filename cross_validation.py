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
from sklearn.metrics import accuracy_score
import nltk
from nltk import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Main Program
train_set = pd.read_csv('train_set.csv', sep="\t", encoding = 'utf8')
train_set = train_set[0:100]

kf = KFold(n_splits=2)
kf.get_n_splits(train_set)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
