import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import nltk
from nltk import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from knearest import euclidean_distance

def write_to_csv(acc, precision, recall, fMeasure):
    acc = ['Accuracy'] + acc
    print acc
    precision = ['Precision'] + precision
    print precision
    recall = ['Recall'] + recall
    print recall
    fMeasure = ['F-Measure'] + fMeasure
    #Titles of the array
    df = pd.DataFrame([acc, precision, recall, fMeasure],columns=['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM'])
    df.to_csv("EvaluationMetric_10fold.csv", sep="\t", index=False)

# Main Program
# Classifiers
random_forest =  RandomForestClassifier()
mult_bayes = MultinomialNB(alpha=0.01)
parameters = {'C': [100],
              'gamma': [0.0001],
              'kernel':['linear','rbf'] }

svc = svm.SVC()
svm = GridSearchCV(svc, parameters)

train_set = pd.read_csv('../train_set.csv', sep="\t", encoding = 'utf8')
train_set_content = train_set['Content']
train_set_categories = train_set['Category']

acc = []
precision = []
recall = []
fMeasure = []

rf_acc = 0.0
mnb_acc = 0.0
svm_acc = 0.0

rf_precision = 0
mnb_precision = 0
svm_precision = 0

rf_recall = 0
mnb_recall = 0
svm_recall = 0

rf_fMeasure = 0
mnb_fMeasure = 0
svm_fMeasure = 0


kf = KFold(n_splits=10)
for train_indexes, test_indexes in kf.split(train_set):
    #print("TRAIN:", train_indexes, "TEST:", test_indexes)
    features_train = [train_set_content[i] for i in train_indexes]
    features_test = [train_set_content[i] for i in test_indexes]
    categories_train = [train_set_categories[i] for i in train_indexes]
    categories_test = [train_set_categories[i] for i in test_indexes]

    # Pipeline for features_train -> Content of every article
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)

    # Select only ten from it
    selector = SelectPercentile(f_classif, percentile = 10)
    selector.fit(features_train_transformed, categories_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()

    #print features_train_transformed
    features_test_transformed = selector.transform(features_test_transformed).toarray()

    #print features_test_transformed
    #random_forest
    random_forest.fit(features_train_transformed, categories_train)
    prediction = random_forest.predict(features_test_transformed)
    rf_acc += accuracy_score(prediction, categories_test)
    rf_precision += precision_score(prediction, categories_test, average="macro")
    rf_recall += recall_score(prediction, categories_test, average="macro")
    rf_fMeasure += f1_score(prediction, categories_test, average='macro')
    print rf_acc
    print "RF"

    #MultinomialNB
    mult_bayes.fit(features_train_transformed, categories_train)
    prediction = mult_bayes.predict(features_test_transformed)
    mnb_acc += accuracy_score(prediction, categories_test)
    mnb_precision += precision_score(prediction, categories_test, average="macro")
    mnb_recall += recall_score(prediction, categories_test, average="macro")
    mnb_fMeasure += f1_score(prediction, categories_test, average='macro')
    print mnb_acc
    print "MB"
    #SVM
    svm.fit(features_train_transformed, categories_train)
    prediction = svm.predict(features_test_transformed)
    svm_acc += accuracy_score(prediction, categories_test)
    svm_precision += precision_score(prediction, categories_test, average="macro")
    svm_recall += recall_score(prediction, categories_test, average="macro")
    svm_fMeasure += f1_score(prediction, categories_test, average='macro')
    print svm_acc
    print "SVM"
#find the accuracy_score
rf_acc = rf_acc / 10
mnb_acc = mnb_acc / 10
svm_acc = svm_acc / 10

acc.append(mnb_acc)
acc.append(rf_acc)
acc.append(svm_acc)

#find the _precision_score
rf_precision = rf_precision / 10
mnb_precision = mnb_precision / 10
svm_precision = svm_precision / 10

precision.append(mnb_precision)
precision.append(rf_precision)
precision.append(svm_precision)

#find the _recall_score
rf_recall = rf_recall / 10
mnb_recall = mnb_recall / 10
svm_recall = svm_recall / 10

recall.append(mnb_recall)
recall.append(rf_recall)
recall.append(svm_recall)

# Find the fMeasure score
rf_fMeasure = rf_fMeasure / 10
mnb_fMeasure = mnb_fMeasure / 10
svm_fMeasure = svm_fMeasure / 10

fMeasure.append(mnb_fMeasure)
fMeasure.append(rf_fMeasure)
fMeasure.append(svm_fMeasure)

write_to_csv(acc, precision, recall, fMeasure)
