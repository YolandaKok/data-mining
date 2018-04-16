import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

<<<<<<< HEAD:svm/10-fold.py
random_forest =  RandomForestClassifier()

def kfold_acc(n_components):
    # Main Program
    # Classifier

    #https://stackoverflow.com/questions/46330329/finding-the-values-of-c-and-gamma-to-optimise-svm
    #You are looking for Hyper-Parameter tuning. In parameter tuning we pass a dictionary containing
    #a list of possible values for you classifier, then depending on the method that
    #you choose (i.e. GridSearchCV, RandomSearch, etc.) the best possible parameters are returned.

=======
def kfold_acc(n_components):
    # Main Program
    # Classifier
    clf = RandomForestClassifier()

>>>>>>> 1524765af8cc994d925f6ea14961e1ff85e844fd:acc-com/acc-components.py
    train_set = pd.read_csv('../train_set.csv', sep="\t", encoding = 'utf8')
    train_set_content = train_set['Content']
    train_set_categories = train_set['Category']

    kf = KFold(n_splits=10)
    for train_indexes, test_indexes in kf.split(train_set):

        features_train = [train_set_content[i] for i in train_indexes]
        features_test = [train_set_content[i] for i in test_indexes]
        categories_train = [train_set_categories[i] for i in train_indexes]
        categories_test = [train_set_categories[i] for i in test_indexes]

        # Pipeline for features_train -> Content of every article
        vectorizer = TfidfVectorizer(stop_words='english')
        features_train_transformed = vectorizer.fit_transform(features_train)
        features_test_transformed = vectorizer.transform(features_test)
<<<<<<< HEAD:svm/10-fold.py
=======

        svd = TruncatedSVD(n_components)
        features_train_transformed = svd.fit_transform(features_train_transformed)
        features_test_transformed = svd.transform(features_test_transformed)
>>>>>>> 1524765af8cc994d925f6ea14961e1ff85e844fd:acc-com/acc-components.py

        # Select only ten from it
        selector = SelectPercentile(f_classif, percentile = 10)
        selector.fit(features_train_transformed, categories_train)
        features_train_transformed = selector.transform(features_train_transformed)

        #print features_train_transformed
        features_test_transformed = selector.transform(features_test_transformed)

        #print features_test_transformed
        #random_forest
        random_forest.fit(features_train_transformed, categories_train)
        prediction = random_forest.predict(features_test_transformed)
        acc = accuracy_score(prediction, categories_test)

    print acc
    return acc


acc = []
<<<<<<< HEAD:svm/10-fold.py
n_components = [1, 50, 100, 150, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
=======
n_components = [10, 20, 50, 100, 150, 200, 500, 600]
>>>>>>> 1524765af8cc994d925f6ea14961e1ff85e844fd:acc-com/acc-components.py

for i in n_components:
    acc.append(kfold_acc(i))

plt.plot(n_components, acc)
plt.axis([0, 600, 0 , 1])
plt.savefig('foo.png')
