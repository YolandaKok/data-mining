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
import nltk
from nltk import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

def write_to_csv(acc, precision):
    acc = ['Accuracy'] + acc
    print acc
    precision = ['Precision'] + precision
    print precision
    #Titles of the array
    df = pd.DataFrame([acc, precision],columns=['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM'])
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
train_set = train_set[0:1000]
train_set_content = train_set['Content']
train_set_categories = train_set['Category']

acc = []
precision = []
recall = []

rf_acc = 0.0
mnb_acc = 0.0
svm_acc = 0.0

rf_precision = 0
mnb_precision = 0
svm_precision = 0

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
    #random_forest
    random_forest.fit(features_train_transformed, categories_train)
    prediction = random_forest.predict(features_test_transformed)
    rf_acc += accuracy_score(prediction, categories_test)
    rf_precision += precision_score(prediction, categories_test, average="macro")

    #MultinomialNB
    mult_bayes.fit(features_train_transformed, categories_train)
    prediction = mult_bayes.predict(features_test_transformed)
    mnb_acc += accuracy_score(prediction, categories_test)
    mnb_precision += precision_score(prediction, categories_test, average="macro")

    #SVM
    svm.fit(features_train_transformed, categories_train)
    prediction = svm.predict(features_test_transformed)
    svm_acc += accuracy_score(prediction, categories_test)
    svm_precision += precision_score(prediction, categories_test, average="macro")

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

write_to_csv(acc, precision)
