import pandas as pd
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
	for filename in filenames:
		print(os.path.join(dirname, filename))

df = pd.read_csv('logistic_regression\heart_disease_data.csv')
print(df.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,0:-1],df.iloc[:,-1],test_size=0.2,random_state=2)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy of Logistic Regression", accuracy_score(y_test, y_pred1))
print("Accuracy of Decision Trees",accuracy_score(y_test,y_pred2))

print(confusion_matrix(y_test, y_pred1))
