# -*- coding: utf-8 -*-
"""
Created on Mon May 17 21:30:44 2021

@author: ismail özdere
"""

# 1. kütüphaneler

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 


# 2 veri önişleme 

# 2,1 verilerin yüklenmesi

veriler = pd.read_csv("Churn_Modelling.csv")

x = veriler.iloc[:,3:13].values # bağımsız
y = veriler.iloc[:,-1:].values # bağımlı 

#encoder 


from sklearn import preprocessing 

le= preprocessing.LabelEncoder()
x[:,1] = le.fit_transform(x[:,1])

le2= preprocessing.LabelEncoder()
x[:,2] = le2.fit_transform(x[:,2])


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float) , [1]) ],
                        remainder="passthrough"
                        )

x = ohe.fit_transform(x)
x = x[:,1:]

# verilerin eğitim ve test için bölünmesi

from sklearn.model_selection import train_test_split 

x_train ,x_test , y_train , y_test = train_test_split( x , y ,test_size = 0.33 , random_state = 0)

# verilerin ölçeklenmesi 

from sklearn.preprocessing import StandardScaler 

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# yapay sinir ağı 

import keras 

from keras.models import Sequential
from keras.layers import Dense

classifier =  Sequential()

classifier.add(Dense(6, input_dim=11, activation='relu'))

classifier.add(Dense(6 , activation='relu'))

classifier.add(Dense(1 , activation='sigmoid'))

classifier.compile(optimizer= "adam" , loss= "binary_crossentropy" , metrics= ["accuracy"] )

classifier.fit(X_train , y_train , epochs=  50  )

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test , y_pred)

print(cm)






















