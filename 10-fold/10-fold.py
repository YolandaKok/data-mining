import pandas as pd

import numpy as np

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import scale
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

import nltk
from nltk.corpus import stopwords
from knearest import findNeighbors
from knearest import predict

def write_to_csv(acc, precision, recall, fMeasure):
    acc = ['Accuracy'] + acc
    precision = ['Precision'] + precision
    recall = ['Recall'] + recall
    fMeasure = ['F-Measure'] + fMeasure
    #Titles of the array
    df = pd.DataFrame([acc, precision, recall, fMeasure],columns=['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM', 'KNN'])
    df.to_csv("EvaluationMetric_10fold.csv", sep="\t", index=False)

# Main Program
# Classifiers
random_forest =  RandomForestClassifier()
mult_bayes = MultinomialNB(alpha=0.01)
parameters = {'C': [100],
              'gamma': [0.0001],
              'kernel':['linear'] }

svc = svm.SVC()
svm = GridSearchCV(svc, parameters)

train_set = pd.read_csv('../train_set.csv', sep="\t", encoding = 'utf8')
train_set_content = train_set['Content'] + 2 * train_set['Title']
train_set_categories = train_set['Category']

acc = []
precision = []
recall = []
fMeasure = []

rf_acc = 0.0
mnb_acc = 0.0
svm_acc = 0.0
knn_acc = 0.0

rf_precision = 0.0
mnb_precision = 0.0
svm_precision = 0.0
knn_precision = 0.0

rf_recall = 0.0
mnb_recall = 0.0
svm_recall = 0.0
knn_recall = 0.0

rf_fMeasure = 0.0
mnb_fMeasure = 0.0
svm_fMeasure = 0.0
knn_fMeasure = 0.0

stop_words = stopwords.words('english')
manual_stop_words = ["said", "say", "want", "one", "know", "two", "see", "something", "also", "says", "get"]

kf = KFold(n_splits=10)
for train_indexes, test_indexes in kf.split(train_set):

    features_train = [train_set_content[i] for i in train_indexes]
    features_test = [train_set_content[i] for i in test_indexes]
    categories_train = [train_set_categories[i] for i in train_indexes]
    categories_test = [train_set_categories[i] for i in test_indexes]

    # Pipeline for features_train -> Content of every article
    stop_words += manual_stop_words
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)

    # lsi
    svd = TruncatedSVD(n_components = 200)
    features_train_lsi = svd.fit_transform(features_train_transformed)
    features_test_lsi = svd.transform(features_test_transformed)

    # Select only ten from it
    selector = SelectPercentile(f_classif, percentile = 10)

    selector.fit(features_train_lsi, categories_train)
    features_train_lsi = selector.transform(features_train_lsi)
    features_test_lsi = selector.transform(features_test_lsi)

    # print features_test_transformed
    # random_forest
    random_forest.fit(features_train_lsi, categories_train)
    prediction = random_forest.predict(features_test_lsi)


    rf_acc += accuracy_score(prediction, categories_test)
    rf_precision += precision_score(prediction, categories_test, average="macro")
    rf_recall += recall_score(prediction, categories_test, average="macro")
    rf_fMeasure += f1_score(prediction, categories_test, average='macro')
    print("rnd", rf_acc)

    #MultinomialNB
    mult_bayes.fit(features_train_transformed, categories_train)
    prediction = mult_bayes.predict(features_test_transformed)

    mnb_acc += accuracy_score(prediction, categories_test)
    mnb_precision += precision_score(prediction, categories_test, average="macro")
    mnb_recall += recall_score(prediction, categories_test, average="macro")
    mnb_fMeasure += f1_score(prediction, categories_test, average='macro')
    print("mnd", mnb_acc)

    #SVM
    svm.fit(features_train_lsi, categories_train)
    prediction = svm.predict(features_test_lsi)

    svm_acc += accuracy_score(prediction, categories_test)
    svm_precision += precision_score(prediction, categories_test, average="macro")
    svm_recall += recall_score(prediction, categories_test, average="macro")
    svm_fMeasure += f1_score(prediction, categories_test, average='macro')
    print("svm", svm_acc)

    # KNN
    prediction = predict(features_train_lsi, features_test_lsi, 5, categories_train)

    knn_acc += accuracy_score(prediction, categories_test)
    knn_precision += precision_score(prediction, categories_test, average="macro")
    knn_recall += recall_score(prediction, categories_test, average="macro")
    knn_fMeasure += f1_score(prediction, categories_test, average='macro')
    print("knn", knn_acc)

#find the accuracy_score
rf_acc = rf_acc / 10
mnb_acc = mnb_acc / 10
svm_acc = svm_acc / 10
knn_acc = knn_acc / 10

acc.append(mnb_acc)
acc.append(rf_acc)
acc.append(svm_acc)
acc.append(knn_acc)

#find the _precision_score
rf_precision = rf_precision / 10
mnb_precision = mnb_precision / 10
svm_precision = svm_precision / 10
knn_precision = knn_precision / 10

precision.append(mnb_precision)
precision.append(rf_precision)
precision.append(svm_precision)
precision.append(knn_precision)

#find the _recall_score
rf_recall = rf_recall / 10
mnb_recall = mnb_recall / 10
svm_recall = svm_recall / 10
knn_recall = knn_recall / 10

recall.append(mnb_recall)
recall.append(rf_recall)
recall.append(svm_recall)
recall.append(knn_recall)

# Find the fMeasure score
rf_fMeasure = rf_fMeasure / 10
mnb_fMeasure = mnb_fMeasure / 10
svm_fMeasure = svm_fMeasure / 10
knn_fMeasure = knn_fMeasure / 10

fMeasure.append(mnb_fMeasure)
fMeasure.append(rf_fMeasure)
fMeasure.append(svm_fMeasure)
fMeasure.append(knn_fMeasure)

write_to_csv(acc, precision, recall, fMeasure)
