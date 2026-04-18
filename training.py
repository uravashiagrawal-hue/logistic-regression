class myLR:
	def __init__(self):
		self.m = None
		self.b = None

	def fit(self,x_train,y_train):
		num =0
		den = 0
		for i in range(x_train.shape[0]):
			num = num+((x_train[i] - x_train.mean())*(y_train[i] - y_train.mean()))
			den = den + ((x_train[i] - x_train.mean())**2)

		self.m = num/den
		self.b= y_train.mean() - (self.m * x_train.mean())

		print(self.m)
		print(self.b)

	def predict(self,x_test):
		return self.b + (self.m * x_test.mean())


import numpy as np
import pandas as pd
df =pd.read_csv('placement.csv')
x= df.iloc[:,0].values
y= df.iloc[:,1].values
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

print(x_train.shape)

lr = myLR()
lr.fit(x_train, y_train)
print(lr.predict(x_test[0]))

