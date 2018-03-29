import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from os import path
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')
# Import data from csv
train_data = pd.read_csv('train_set.csv', sep="\t", encoding = 'utf8')
politics = train_data[train_data['Category'] == 'Politics']
politics = politics['Content']
#print type(politics)

A = np.array(politics)
#print A.shape

#print type(stra)
stop_words = stopwords.words('english')
stop_words.append("said")
stop_words.append("want")

#filtered_sentence = [w for w in A[0] if not w in stop_words]

filtered_sentence = []
word = []

for w in A[0]:
    if w != ' ':
        word.append(w)
    else:
        word = ''.join(word)
        if word not in stop_words:
            filtered_sentence.append(word)
        word = []

filtered_sentence = ' '.join(filtered_sentence)
#print filtered_sentence

#wordcloud = WordCloud().generate(filtered_sentence)

# Display the generated image:
# the matplotlib way:

#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(filtered_sentence)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
