import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import math
from pandas import DataFrame

# Euclidean Distance
def euclidean_distance(vector1, vector2):
    length1 = len(vector1)
    length2 = len(vector2)
    # Make vectors have the same length
    if (length1 > length2):
        vector1 = vector1[:length2]
    elif (length2 > length1):
        vector2 = vector2[:length1]

    distance = 0.0
    # Calculate the euclidean_distance
    for i in range(len(vector1)):
        distance += pow((vector1[i] - vector2[i]), 2)
    return math.sqrt(distance)

# Find the K nearest neighbors
def findNeighbors(trainData, testItem, k):
    # Calculate the nearest neighbors of the training Data
    distances = []


a = [2, 9, 8, 5]
b = [2, 1, 2, 6]
print euclidean_distance(a, b)

# Read the train_set
train_set = pd.read_csv('train_set.csv', sep="\t", encoding = 'utf8')
train_content = train_set['Content']
# Read the test_set
test_set = pd.read_csv('test_set.csv', sep="\t", encoding = 'utf8')
test_content = test_set['Content']

vectorizer = CountVectorizer(stop_words='english')

transformed = vectorizer.fit_transform(train_content[:1])
print DataFrame(transformed.A[0]).values.astype(int).tolist()
#print transformed
