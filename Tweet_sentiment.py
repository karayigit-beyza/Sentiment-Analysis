# Natural Language Processing 

#Importing the Libraries 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 


#Importing the datasets 
testdata = pd.read_csv('Trump.txt', delimiter = '\t', names=["a", "text", "c", "d", "e", "f", "g"], quoting = 3, nrows=3000)
traindata = pd.read_csv('train_twt.csv', delimiter = ',', encoding="ISO-8859-1", names=["label", "b", "c", "d", "e", "f", "tweets"], quoting = 3, error_bad_lines=False)

traindata.head(10)



#Checking the training dataset column names and values 
traindata['label'].value_counts()
#traindata.columns
#traindata.shape


#Creating subset training data 
traindata = traindata.drop(["b", "c", "d", "e", "f"], axis = 1)
traindata.dropna(subset=['tweets'], inplace=True)
traindata = traindata.sample(frac = 0.4, random_state = 42)
traindata['label'].value_counts()



#selectedone = np.random.choice(366850, replace=False, size=150000)
#traindata = traindata.iloc[selectedone]
#traindata.groupby('label').count()

#Fixing Index
min(traindata.index)
max(traindata.index)
traindata.head()
traindata = traindata.reset_index().drop(columns = 'index')


#Cleaning the texts for training data
#traindata = traindata.drop(["a", "c", "d", "e", "f", "g"], axis = 1)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

corpus_train = []


for i in range(0,146740):
    text =re.sub('[^a-zA-Z]', ' ', traindata['tweets'][i] )
    text = text.lower()
    text  = text.split()
    ps = PorterStemmer()
    text  = [ps.stem(word) for word in text  if not word in set(stopwords.words('english'))]
    text  = ' '.join(text)
    corpus_train.append(text)
    


#Cleaning the text for test data 
corpus_test = []
for i in range(0,testdata.shape[0]):
    text_test =re.sub('[^a-zA-Z]', ' ', testdata['text'][i] )
    text_test = text_test.lower()
    text_test = text_test.split()
    ps = PorterStemmer()
    text_test  = [ps.stem(worda) for worda in text_test  if not worda in set(stopwords.words('english'))]
    text_test  = ' '.join(text_test)
    corpus_test.append(text_test)   

    
#Creating the Bag of Words Model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 6000)

cv_fit = cv.fit(corpus_train)
X_train = cv_fit.transform(corpus_train).toarray()
y_train = traindata.iloc[:, 0].values

cv_test = CountVectorizer(max_features = 6000)
X_test = cv_fit.transform(corpus_test).toarray()


# Fitting the Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from collections import Counter
Counter(y_pred)

1839/(1839+1161)

##%61 percent The President Trump Talks Negative 