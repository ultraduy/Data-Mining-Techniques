import pandas as pd
import numpy as np
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
import os
import string
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

data = pd.read_csv('SmsCollection.csv', delimiter='m;', skiprows=1, header=None)
split = 4459 
train_data = data[:split]
test_data = data[split:]

# In[] Data preparation
data = train_data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer() 
translate_table = dict((ord(char), None) for char in string.punctuation)   

dictionaryHam = {}
dictionarySpam = {}
ham = []
spam = []

h = 0
s = 0
t = 0
for message in data[1]:
# Removal of stop words
    word_tokens = [w.lower() for w in word_tokenize(message)] # make all letters lowercase
    filtered_tokens = [w for w in word_tokens if not w.lower() in stop_words]
    
    if(data[0][t]=='ha'):
        typeData = 'ham'
    else:
        typeData = 'spam'
    
# Lemmatization
    new_filtered_tokens = []
    for i in np.arange(0,len(filtered_tokens)):
        word = filtered_tokens[i].translate(translate_table)    # remove non-words
        if(len(word) is not 1):                                 # remove single characters
            lemmatized_token = lemmatizer.lemmatize(word)
            if(lemmatized_token is not ''):
                filtered_tokens[i] = lemmatized_token
                if(typeData == 'ham'):
                    ham.append(lemmatized_token)
                else:
                    spam.append(lemmatized_token)
        
    if(data[0][t]=='ha'):
        dictionaryHam[t] = filtered_tokens
        h+=1
    else:
        dictionarySpam[t] = filtered_tokens
        s+=1
    t+=1
    
fdist_ham = FreqDist(ham)
fdist_spam = FreqDist(spam)

most_common_ham = fdist_ham.most_common(20)
most_common_spam = fdist_spam.most_common(20)
print('Most common words in ham messages: \n', most_common_ham)
print('Most common words in spam messages: \n', most_common_spam)

  
