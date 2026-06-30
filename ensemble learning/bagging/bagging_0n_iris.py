import numpy as np
import pandas as pd
df = pd.read_csv('ensemble learning\iris.csv')
print(df.head())
df = df.iloc[:,1:]
print(df.head())
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Species']= encoder.fit_transform(df['Species'])
print(df.head())

df = df[df['Species'] != 0][['SepalWidthCm', 'PetalLengthCm','Species']]
print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt
plt.scatter(df['SepalWidthCm'],df['PetalLengthCm'],c=df['Species'],cmap='winter')
plt.show()

df_train = df.iloc[:60,:].sample(10)
print(df_train)

df = df.sample(100)
df_train = df.iloc[:60,:].sample(10)
df_val = df.iloc[60:80,:].sample(5)
df_test = df.iloc[80:,:].sample(5)

X_test = df_val.iloc[:,0:2].values
y_test = df_val.iloc[:,-1].values
print(y_test)


# BAGGING
df_bag = df_train.sample(8,replace=True)

X = df_bag.iloc[:,0:2]
y = df_bag.iloc[:,-1]
print(df_bag)

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score

def evaluate(clf,X,y):
    clf.fit(X,y)
    plot_tree(clf)
    plt.show()
    plot_decision_regions(X.values, y.values, clf=clf, legend=2)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test,y_pred))

dt_bag1 =DecisionTreeClassifier()
evaluate(dt_bag1,X,y)

# Data for Tree 1
df_bag = df_train.sample(8,replace=True)

# Fetch X and y
X = df_bag.iloc[:,0:2]
y = df_bag.iloc[:,-1]

# print df_bag
df_bag
dt_bag2 = DecisionTreeClassifier()
evaluate(dt_bag2,X,y)
