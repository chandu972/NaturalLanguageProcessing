import re # for regular expressions
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk # for text manipulation
import warnings 
from nltk.corpus import stopwords
stop = stopwords.words('english')
from textblob import TextBlob

#Read the twitter sentiment dataset
train = pd.read_csv('G:/DataScienceLearning/train_E6oV3lV.csv')

#No. of words 
#Extract the number of words in each tweet. The basic intuition behind this is that generally, 
#the negative sentiments contain a lesser amount of words than the positive ones
train['word_count'] = train['tweet'].apply(lambda x: len(str(x).split(" ")))
train[['tweet','word_count']].head()

#Extract the no. of characters in tweet
train['char_count'] = train['tweet'].str.len() ## this also includes spaces
train[['tweet','char_count']].head()

#calculate the average word length of each tweet.
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

train['avg_word'] = train['tweet'].apply(lambda x: avg_word(x))

train[['tweet','avg_word']].head()

#Import stopwords
train['stopwords'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
train[['tweet','stopwords']].head()

#No. of special characters (Extract hashtags)
train['hastags'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['tweet','hastags']].head()

#No. of numerics
train['numerics'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['tweet','numerics']].head()

#No. of uppercase
train['upper'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['tweet','upper']].head()

#No. of lowercase
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['tweet'].head()

#Remove punctuation
train['tweet'] = train['tweet'].str.replace('[^\w\s]','')
train['tweet'].head()

#Removal of stop words : stop words (or commonly occurring words) should be removed from the text data
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['tweet'].head()

#remove commonly occurring words from our text data First, 
#let’s check the 10 most frequently occurring words in our text data then take call to remove or retain.
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]
freq

freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()

#this time let’s remove rarely occurring words from the text. 
#Because they’re so rare, the association between them and other words is dominated by noise. 
#You can replace rare words with a more general form and then this will have higher counts
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
freq

freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()

#Spelling correction
train['tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))

#Tokenization refers to dividing the text into a sequence of words or sentences. In our example,
#we have used the textblob library to first transform our tweets into a blob and then converted them into a series of words.
TextBlob(train['tweet'][1]).words

#Stemming refers to the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. 
#For this purpose, we will use PorterStemmer from the NLTK library.
from nltk.stem import PorterStemmer
st = PorterStemmer()
train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

#Lemmatization is a more effective option than stemming because it converts the word into its root word, 
#rather than just stripping the suffices. It makes use of the vocabulary and does a morphological analysis 
#to obtain the root word. Therefore, we usually prefer using lemmatization over stemming.
from textblob import Word
train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train['tweet'].head()

#extracting features using NLP techniques.
#NGrams
#N-grams are the combination of multiple words used together. 
#Ngrams with N=1 are called unigrams. Similarly, bigrams (N=2), trigrams (N=3) and so on can also be used.

#Unigrams do not usually contain as much information as compared to bigrams and trigrams.
#The basic principle behind n-grams is that they capture the language structure, 
#like what letter or word is likely to follow the given one. 
#The longer the n-gram (the higher the n), the more context you have to work with. 

TextBlob(train['tweet'][0]).ngrams(2)

#Term frequency is simply the ratio of the count of a word present in a sentence, to the length of the sentence.

#Therefore, we can generalize term frequency as:

#TF = (Number of times term T appears in the particular row) / (number of terms in that row)

tf1 = (train['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1

#The intuition behind inverse document frequency (IDF) is that a word is not of much use to us
#if it’s appearing in all the documents.

#Therefore, the IDF of each word is the log of the ratio of the total number of rows to the number of rows
#in which that word is present.

#IDF = log(N/n), where, N is the total number of rows and n is the number of rows in which the word was present.

#So, let’s calculate IDF for the same tweets for which we calculated the term frequency.

for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['tweet'].str.contains(word)])))

tf1

#TF-IDF is the multiplication of the TF and IDF which we calculated above
tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1

#We don’t have to calculate TF and IDF every time beforehand and then multiply it to obtain TF-IDF. 
#Instead, sklearn has a separate function to directly obtain it:
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(train['tweet'])
print(train_vect)

#Bag of Words (BoW) refers to the representation of text which describes the presence of words within the text data.
#The intuition behind this is that two similar text fields will contain similar kind of words, 
#and will therefore have a similar bag of words.
#Further, that from the text alone we can learn something about the meaning of the document.
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(train['tweet'])
train_bow

#If you recall, our problem was to detect the sentiment of the tweet. 
#So, before applying any ML/DL models (which can have a separate feature detecting the sentiment using the textblob library),
#let’s check the sentiment of the first few tweets.

train['tweet'][:5].apply(lambda x: TextBlob(x).sentiment)

#Above, you can see that it returns a tuple representing polarity and subjectivity of each tweet.
#Here, we only extract polarity as it indicates the sentiment as value nearer to 1 means a positive sentiment 
#and values nearer to -1 means a negative sentiment.
#This can also work as a feature for building a machine learning model.

train['sentiment'] = train['tweet'].apply(lambda x: TextBlob(x).sentiment[0] )
train[['tweet','sentiment']].head()
