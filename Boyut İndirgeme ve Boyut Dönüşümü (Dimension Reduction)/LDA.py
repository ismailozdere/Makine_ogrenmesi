# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:27:57 2021

@author: ismail özdere
"""

# kütüphaneler 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# veri kümesi 

veriler = pd.read_csv("Wine.csv")

X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values

# eğitim ve test için kümelerin bölünmesi

from sklearn.model_selection import train_test_split

X_train ,X_test ,y_train , y_test = train_test_split(X, y , test_size = 0.2 , random_state = 0 )


#Ölçekleme 

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# PCA

from sklearn.decomposition import PCA

pca = PCA(n_components= 2 )

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)


# pca dönüşümünden önce gelen LR
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0 )

classifier.fit(X_train, y_train)

# pca dönüşümünden sonra gelen LR
clasifier2 = LogisticRegression(random_state= 0)
clasifier2.fit(X_train2 , y_train)


#tahminler
y_pred = classifier.predict(X_test)

y_pred2 = clasifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix


# actual / PCA olmadan çıkan sonuç
print("gerçek ve PCA olmadan olan")
cm = confusion_matrix(y_test , y_pred)
print(cm)

# actual / PCA sonrası çıkan sonuç
print("gerçek / PCA ile")
cm2 = confusion_matrix(y_test , y_pred2)
print(cm2)

#  PCA sonrası / PCA öncesi
print("PCA'sız PCA'li")
cm3 = confusion_matrix(y_pred , y_pred2)
print(cm3)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components= 2)


X_train_lda =  lda.fit_transform(X_train ,y_train)

X_test_lda = lda.transform((X_test))

#LDA dönüşümünden sonra
clasifier_lda = LogisticRegression(random_state=0)
clasifier_lda.fit(X_train_lda , y_train)


#LDA verisini tahmin et
y_pred_lda = clasifier_lda.predict(X_test_lda)


#  LDA sonrası / orjinal
print("LDA ve orjinal")
cm4 = confusion_matrix(y_pred , y_pred_lda)
print(cm4)

from xgboost import XGBClassifier

clasifier = XGBClassifier()
clasifier.fit(X_train, y_train)


y_pred = clasifier.predict(X_test)
print(y_pred)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred , y_test)

print(cm )













