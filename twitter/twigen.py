
# coding: utf-8

# In[21]:

import pandas as pd
import numpy as np
import re
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
get_ipython().magic('matplotlib inline')
from collections import defaultdict
from collections import Counter
from nltk.tokenize import word_tokenize,wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
import matplotlib.pylab as plt
import string


# In[2]:

check= pd.read_csv("C:/Users/Archit Jhingan/Desktop/twitter-user-gender-classification/gender-classifier-DFE-791531.csv", encoding='latin1')


# In[3]:

check.head()


# In[4]:

check.columns


# In[5]:

user = pd.read_csv("C:/Users/Archit Jhingan/Desktop/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",usecols= [0,5,6,19,17,21,10,11],encoding='latin1')


# In[6]:

user.head()


# In[65]:

def cleaning(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    s = s.replace("ù"," ")
    s = s.replace(" ù] "," ")
    s = s.replace("ü"," ")
    s = s.replace("like"," ")
    return s

user['text_norm'] = [cleaning(s) for s in user['text']]
user['Description'] = [cleaning(s) for s in user['description']]
useless = stopwords.words('english') + list(string.punctuation)
user['text_norm'] = user['text_norm'].apply(wordpunct_tokenize)
user['text_norm'] = user['text_norm'].apply(lambda x : [item for item in x if item not in useless])


# In[64]:

user.head()


# In[46]:

user.gender.value_counts()


# In[47]:

data = user[user['gender:confidence']==1]


# In[48]:

data.head()


# In[49]:

Male = data[data['gender'] == 'male']
Female = data[data['gender'] == 'female']
Brand = data[data['gender'] == 'brand']


# In[50]:

Male_Words = pd.Series(' '.join(Male['text_norm'].astype(str)).lower().split(" ")).value_counts()[:20] 
Female_Words = pd.Series(' '.join(Female['text_norm'].astype(str)).lower().split(" ")).value_counts()[:20]


# In[51]:

Male_Words.plot(kind='bar')


# In[52]:

Female_Words.plot(kind='bar')


# In[16]:

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data['text_norm'].astype(str))

encoder = LabelEncoder()
y = encoder.fit_transform(data['gender'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[17]:

logistic = LogisticRegression()
logistic.fit(x_train,y_train)


# In[18]:

print(logistic.score(x_test, y_test))

