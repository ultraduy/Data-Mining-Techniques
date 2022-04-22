import pandas as pd
import numpy as np
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
import string
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter


data = pd.read_csv('SmsCollection.csv', delimiter='m;', skiprows=1, header=None)
split = 4459 
train_data = data[:split]
test_data = data[split:]

# In[] Data preparation train
data = train_data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer() 
translate_table = dict((ord(char), None) for char in string.punctuation)   

dictionaryHam = {}
dictionarySpam = {}
ham = []
spam = []
all_words_train = []

h = 0
s = 0
t = 0

train = []
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
                all_words_train.append(lemmatized_token)
                if(typeData == 'ham'):
                    ham.append(lemmatized_token)
                    train.append((message,'ham'))
                else:
                    spam.append(lemmatized_token)
                    train.append((message,'spam'))
        
    if(data[0][t]=='ha'):
        dictionaryHam[t] = filtered_tokens
        h+=1
    else:
        dictionarySpam[t] = filtered_tokens
        s+=1
    t+=1
    
fdist_ham = FreqDist(ham)
fdist_spam = FreqDist(spam)
fdist_tot = FreqDist(ham+spam)

most_common_ham = fdist_ham.most_common(1000)
most_common_spam = fdist_spam.most_common(1000)
      
# In[] Predictive Modelling
dict_word_occurrences = {}
for token in most_common_ham:
    word = token[0]
    occurrence = token[1]
    dict_word_occurrences[word] = [occurrence]

words_spam = []    
for token in most_common_spam:
    word = token[0]
    occurrence = token[1]
    if(word in dict_word_occurrences):
        dict_word_occurrences[word].append(occurrence)
    else:
        dict_word_occurrences[word] = [0,occurrence]    
    words_spam.append(word)
        
for word in dict_word_occurrences:
    if(word not in words_spam):
        dict_word_occurrences[word] = [dict_word_occurrences[word][0],0]

# In[] Test set
data = test_data

ham_or_spam_pred = []

for message in data[1]:
# Removal of stop words
    word_tokens = [w.lower() for w in word_tokenize(message)] # make all letters lowercase
    filtered_tokens = [w for w in word_tokens if not w.lower() in stop_words]
    
# Lemmatization
    hams_or_spams = []
    for i in np.arange(0,len(filtered_tokens)):
        word = filtered_tokens[i].translate(translate_table)    # remove non-words
        if(len(word) != 1):                                 # remove single characters
            lemmatized_token = lemmatizer.lemmatize(word)
            if(lemmatized_token != ''):
                if(lemmatized_token in dict_word_occurrences):
                    hams = dict_word_occurrences[lemmatized_token][0]
                    spams = dict_word_occurrences[lemmatized_token][1]
                    if(hams > spams):
                        hams_or_spams.append('ham')
                    else:
                        hams_or_spams.append('spam')
        
    counts = Counter(hams_or_spams)                
    counts_ham = counts['ham']
    counts_spam = counts['spam']
    if(counts_ham >= counts_spam):
        ham_or_spam_pred.append('ham')
    else:
        ham_or_spam_pred.append('spam')

# In[] Quality
pred = ham_or_spam_pred
real = data[0].reset_index(drop=True)

pred_qual = [0,0]        # correctly predicted, false
spam_qual = [0,0]
for i in np.arange(0,len(pred)):
    if(real[i] == 'ha'):
        if(pred[i] == 'ham'):
            pred_qual[0] +=1 
        else:
            pred_qual[1] +=1
            
    else:
        if(pred[i] == 'spam'):
            spam_qual[0] +=1
        else:
            spam_qual[1] +=1

print(pred_qual)
print(spam_qual)