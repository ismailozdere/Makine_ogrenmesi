import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# **2. veri önişleme**

#**2.1 verilerin yüklenmesi**

veriler = pd.read_csv("satislar.csv")

#test

print(veriler)

# **veri önişleme

aylar = veriler[["Aylar"]]
print(aylar)

satislar = veriler[["Satislar"]]
print(satislar)

# **verilerin eğitim ve test için bölünmesi**

from sklearn.model_selection import train_test_split

x_train , x_test ,y_train , y_test = train_test_split(aylar , satislar , test_size= 0.33 , random_state = 0)

# **verilerin ölçeklenmesi**

#from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()

"""X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

**model inşaası ( linear regression)**
"""

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)
print(tahmin)

x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))

