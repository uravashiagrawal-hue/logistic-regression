# KNN - k nearest neighbour

import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\Lenovo\regresion\KNN\data.csv')
print(df.head())
df.drop(columns= ['id','Unnamed: 32'], inplace=True)
print(df.head())
print(df.shape)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.iloc[:,1:], df.iloc[:,0], test_size=0.2, random_state=2)
print(x_train.shape)

# when we work with knn it is adviced ki app apna complete data same scale pr le aao
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
# how we transform - first we find std, and mean of the data, and for each value of data we do (value-mean)/std
# resultant will bw our final data

x_test = scaler.transform(x_test)     #here we don't apply fit as it is calculated in x_train, we will go according to that
print(x_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
# we have to give no of neighbours, hm litne neighbour lena chate hai

knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

scores= []
for i in range(1,16):
	knn2 = KNeighborsClassifier(n_neighbors=1)
	knn2.fit(x_train, y_train)
	y_pred2 = knn2.predict(x_test)
	scores.append(accuracy_score(y_test, y_pred2))

import matplotlib.pyplot as plt
plt.plot(range(1,16), scores)
plt.show()


# with higher value of k it happens under fitting an dfor lower value it happens overfitting

# where knn fails-
# with large datasets - like where we have x=5lakh and features = 100
# knn is the lasy learning technique - where sara kam prediction phase m hota hai
# isme traing phase m kuch nhi hota kewal data collect ho rha hai

# another region is outliers

# non-homogeneous scales

# imbalanced dataset  if we have 98% yes(blue) and 2% no(pink) this will be biased towards blue

#  not good in inference

