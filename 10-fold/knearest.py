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
from sklearn import preprocessing

def write_to_csv(predictions):
    # Transform list of tuples to a dataframe
    df = pd.DataFrame(predictions, columns=['ID', 'Predicted_Category'])
    # Do not include index column
    df.to_csv("testSet_categories.csv", sep="\t", index=False)

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
def findNeighbors(trainData, testItem, k, categories, ids):
    # Calculate the nearest neighbors of the training Data
    distances = []
    # Calculate distance between every element of the trainData to testItem
    n = 0
    for x in trainData:
        distances.append((euclidean_distance(x, testItem), categories[n], ids[n]))
        n += 1
    distances = sorted(distances,  key=lambda x: x[0])[:k]
    return distances

def MajorityVoting(neighbors):
    # Select Neighbors categories
    categories = [0, 0, 0, 0, 0]
    for item in neighbors:
        if item[1] == 'Business':
            categories[0] += 1
        elif item[1] == 'Football':
            categories[1] += 1
        elif item[1] == 'Politics':
            categories[2] += 1
        elif item[1] == 'Film':
            categories[3] += 1
        elif item[1] == 'Technology':
            categories[4] += 1

    print categories
    category_index = categories.index(max(categories))
    return category_index

# Read the train_set
"""train_set = pd.read_csv('train_set.csv', sep="\t", encoding = 'utf8')
train_content = train_set['Title']
# Keep the train_set Categories
train_categories = train_set['Category']
train_id = train_set['Id']
# Read the test_set
test_set = pd.read_csv('test_set.csv', sep="\t", encoding = 'utf8')
test_content = test_set['Title']
# Id of the test set
test_id = test_set['Id']

vectorizer = CountVectorizer(stop_words='english', max_features = 200)

# Train content list of lists
transformed = vectorizer.fit_transform(train_content)
#train_set_list = np.array(transformed)
#print train_set_list
train_set_list = DataFrame(transformed.A).values.astype(int).tolist()
# Test Content list of lists
transformed_test = vectorizer.fit_transform(test_content)
#test_set_list = np.array(transformed_test)
test_set_list = DataFrame(transformed_test.A).values.astype(int).tolist()


neighbors = []
categories = ['Business', 'Football', 'Politics', 'Film', 'Technology']
predictions = []
results = []
for item in test_set_list:
    # Find neighbors for each element
    neighbors = findNeighbors(train_set_list, item, 10, train_categories, train_id)
    # Majority Voting
    result = MajorityVoting(neighbors)
    results.append(categories[result])
predictions = zip(test_id, results)
write_to_csv(predictions)


# Write to csv predicted categories
"""
