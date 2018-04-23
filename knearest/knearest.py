import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import text
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import math
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import scipy


def write_to_csv(predictions):
    # Transform list of tuples to a dataframe
    df = pd.DataFrame(predictions, columns=['ID', 'Predicted_Category'])
    # Do not include index column
    df.to_csv("testSet_categories.csv", sep="\t", index=False)

# Find the K nearest neighbors
def findNeighbors(trainData, testData, k):
    # Calculate the nearest neighbors of the training Data
    indexes = []
    for x in testData:
        # Calculates distances between a test element and all the other elements
        # of the train set
        distance = scipy.spatial.distance.cdist(trainData, [x], 'euclidean')
        # Find the k nearest for the current element
        index = np.argsort(distance[:, 0])[:k]
        indexes.append(index)
    return indexes

def MajorityVoting(neighbors):
    # Select Neighbors categories
    categories = [0, 0, 0, 0, 0]
    for item in neighbors:
        if item == 'Business':
            categories[0] += 1
        elif item == 'Football':
            categories[1] += 1
        elif item == 'Politics':
            categories[2] += 1
        elif item == 'Film':
            categories[3] += 1
        elif item == 'Technology':
            categories[4] += 1

    category_index = categories.index(max(categories))
    return category_index

# Predict Function
# Returns list with the predicted categories (Strings)
def predict(train_set_list, test_set_list, k):
    neighbors = []
    categories = ['Business', 'Football', 'Politics', 'Film', 'Technology']
    predictions = []
    results = []

    indexes = findNeighbors(train_set_list, test_set_list, k)

    for index in indexes:
        for item in index:
            neighbors.append(train_categories[item])
        # Send list of Categories
        # MajorityVoting
        result = MajorityVoting(neighbors)
        results.append(categories[result])
        neighbors = []

    return results

# Read the train_set
train_set = pd.read_csv('../train_set.csv', sep="\t", encoding = 'utf8')
train_content = train_set['Content']
# Keep the train_set Categories
train_categories = train_set['Category']
train_id = train_set['Id']
# Read the test_set
test_set = pd.read_csv('../test_set.csv', sep="\t", encoding = 'utf8')
test_content = test_set['Content']
# Id of the test set
test_id = test_set['Id']

pipeline = Pipeline([
    ('vec', CountVectorizer(max_features=4000, stop_words='english')),
    ('transformer', TfidfTransformer()),
    ('svd', TruncatedSVD(n_components=30))
])

# Train content list of lists
train_set_list = pipeline.fit_transform(train_content)
test_set_list = pipeline.fit_transform(test_content)

results = predict(train_set_list, test_set_list, 3)

predictions = zip(test_id, results)
# Write to csv predicted categories
write_to_csv(predictions)
