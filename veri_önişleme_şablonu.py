
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv("satislar.csv")


aylar = veriler[["Aylar"]]
print(aylar)

satislar = veriler[["Satislar"]]
print(satislar)



from sklearn.model_selection import train_test_split

x_train , x_test ,y_train , y_test = train_test_split(aylar , satislar , test_size= 0.33 , random_state = 0)

"""**verilerin ölçeklenmesi**"""

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

