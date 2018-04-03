import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from os import path
from wordcloud import WordCloud

#nltk.download('stopwords')
#nltk.download('punkt')


# Import data from csv
train_data = pd.read_csv('train_set.csv', sep="\t", encoding = 'utf8')

politics = train_data[train_data['Category'] == 'Politics']
politics = politics['Content']

football = train_data[train_data['Category'] == 'Football']
football = football['Content']

film = train_data[train_data['Category'] == 'Film']
film = film['Content']

technology = train_data[train_data['Category'] == 'Technology']
technology = technology['Content']

business = train_data[train_data['Category'] == 'Business']
business = business['Content']


stop_words = stopwords.words('english')
manual_stop_words = ["said", "say", "want", "one", "know", "two", 
                     "see", "something", "also", "says", "get"]


#make the content of the politics and make it into a text
A = np.array(politics)

filtered_sentence = []
word = []

for i in range(A.shape[0]):
  for w in A[i]:
      if w != ' ':
          word.append(w)
      else:
          word = ''.join(word)
          #if word not in stop_words and word not in manual_stop_words:
          filtered_sentence.append(word)
          word = []

filtered_sentence = ' '.join(filtered_sentence)
wordcloud = WordCloud(stopwords=stop_words + manual_stop_words).generate(filtered_sentence)
plt.imsave("./images/politics.png", wordcloud)

#films
A = np.array(film)

filtered_sentence = []
word = []

for i in range(A.shape[0]):
  for w in A[i]:
      if w != ' ':
          word.append(w)
      else:
          word = ''.join(word)
          filtered_sentence.append(word)
          word = []

filtered_sentence = ' '.join(filtered_sentence)

wordcloud = WordCloud(stopwords=stop_words + manual_stop_words).generate(filtered_sentence)
plt.imsave("./images/film.png", wordcloud)

#football
A = np.array(football)

filtered_sentence = []
word = []

for i in range(A.shape[0]):
  for w in A[i]:
      if w != ' ':
          word.append(w)
      else:
          word = ''.join(word)
          filtered_sentence.append(word)
          word = []

filtered_sentence = ' '.join(filtered_sentence)

wordcloud = WordCloud(stopwords=stop_words + manual_stop_words).generate(filtered_sentence)
plt.imsave("./images/football.png", wordcloud)

#technology
A = np.array(technology)

filtered_sentence = []
word = []

for i in range(A.shape[0]):
  for w in A[i]:
      if w != ' ':
          word.append(w)
      else:
          word = ''.join(word)
          filtered_sentence.append(word)
          word = []

filtered_sentence = ' '.join(filtered_sentence)

wordcloud = WordCloud(stopwords=stop_words + manual_stop_words).generate(filtered_sentence)
plt.imsave("./images/technology.png", wordcloud)

#business
A = np.array(politics)

filtered_sentence = []
word = []

for i in range(A.shape[0]):
  for w in A[i]:
      if w != ' ':
          word.append(w)
      else:
          word = ''.join(word)
          filtered_sentence.append(word)
          word = []

filtered_sentence = ' '.join(filtered_sentence)

wordcloud = WordCloud(stopwords=stop_words + manual_stop_words).generate(filtered_sentence)
plt.imsave("./images/bussiness.png", wordcloud)
