# making our own linear regression class

import numpy as np

class MeraLR:
	def __init__(self):
		self.coef_ = None
		self.intercept_ = None

	def fit(self,x_train,y_train):
		x_train = np.insert(x_train,0,1,axis=1)
		# adding a column of all 1 in the the matrix of X, check in copy

		# calculate the coef
		betas = np.linalg.inv(np.dot(x_train.T,x_train)).dot(x_train.T).dot(y_train)
		self.intercept_= betas[0]
		self.coef_ = betas[1:]

	def predict(self,x_test):
		y_pred = np.dot(x_test,self.coef_) + self.intercept_
		return y_pred

from sklearn.datasets import load_diabetes
x,y = load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.2,random_state=3)

lr = MeraLR()
lr.fit(x_train,y_train)
print(x_train.shape)
print(np.insert(x_train,0,1,axis=1).shape)

y_pred = lr.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

print(lr.coef_)
print(lr.intercept_)


