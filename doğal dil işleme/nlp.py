#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:00:54 2021

@author: ismail özdere
"""

import numpy as np
import pandas as pd

yorumlar = pd.read_csv('restorant_yorumları.csv')

import re
import nltk

yorumlar = yorumlar.dropna()

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

#Preprocessing (Önişleme)
derlem = []

for i in range(1000):

    try: 
        
        yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
        yorum = yorum.lower()
        yorum = yorum.split()
        yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
        yorum = ' '.join(yorum)
        derlem.append(yorum)

    except:
        continue


#Feautre Extraction ( Öznitelik Çıkarımı)
#Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(derlem).toarray() # bağımsız değişken
y = yorumlar.iloc[:,1].values # bağımlı değişken

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)



















