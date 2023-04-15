### opis:wstęp o fake news, jak działa machine learning, jak trenuje sie kod, testuje sie dane
    # kod: kod, szukanie słów(?), przygotowanie danych/tekstow, podzia na data for training i data for testing

#START python:

using Pkg
Pkg.add("Python.call")
using PythonCall

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.preprocessing.text import one_hot

tkl = pyimport("tensorflow.keras.layers")
pad_sequences = pyimport("pad_sequences")
tkm = pyimport("tensorflow.keras.models")
Sequential = pyimport("Sequential")
tkpt = pyimpory("tensorflow.keras.preprocessing.text")
one_hot = pyimport("one_hot")

df = pd.read_csv('../input/fake-news/train.csv')
test = pd.read_csv('../input/fake-news/test.csv')

df.head()

#filling NULL values with empty string
df=df.fillna('')
test=test.fillna('')

# We will be only using title and author name for prediction
# Creating new coolumn total concatenating title and author
df['total'] = df['title']+' '+df['author']
test['total']=test['title']+' '+test['author']


X = df.drop('label',axis=1)
y=df['label']
print(X.shape)
print(y.shape)

#Choosing vocabulary size to be 5000 and copying data to msg for further cleaning
voc_size = 5000
msg = X.copy()
msg_test = test.copy()


#Downloading stopwords 
#Stopwords are the words in any language which does not add much meaning to a sentence.
#They can safely be ignored without sacrificing the meaning of the sentence.
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

#We will be using Stemming here
#Stemming map words to their root forms
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []


#Applying stemming and some preprocessing
for i in range(len(msg)):
  review = re.sub('[^a-zA-Z]',' ',msg['total'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus.append(review)

  #Applying stemming and some preprocessing for test data
corpus_test = []
for i in range(len(msg_test)):
  review = re.sub('[^a-zA-Z]',' ',msg_test['total'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus_test.append(review)


# Converting to one hot representation
onehot_rep = [one_hot(words,voc_size)for words in corpus]
onehot_rep_test = [one_hot(words,voc_size)for words in corpus_test]


#Padding Sentences to make them of same size
embedded_docs = pad_sequences(onehot_rep,padding='pre',maxlen=25)
embedded_docs_test = pad_sequences(onehot_rep_test,padding='pre',maxlen=25)

#We have used embedding layers with LSTM
model = Sequential()
model.add(Embedding(voc_size,40,input_length=25))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

#Converting into numpy array
X_final = np.array(embedded_docs)
y_final = np.array(y)
test_final = np.array(embedded_docs_test)
X_final.shape,y_final.shape,test_final.shape

model.fit(X_final,y_final,epochs=20,batch_size=64)

y_pred = model.predict_classes(test_final)

final_sub = pd.DataFrame()
final_sub['id']=test['id']
final_sub['label'] = y_pred
final_sub.to_csv('final_sub.csv',index=False)


final_sub.head()

