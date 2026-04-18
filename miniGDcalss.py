from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time
import random

x,y = load_diabetes(return_X_y=True)
print(x.shape)
print(y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=2)

class MBGDregressor:
	def __init__(self,batch_size,learning_rate = 0.1, epochs = 100):
		self.coef_ = None
		self.intercept_ = None
		self.lr = learning_rate
		self.epochs= epochs
		self.batch_size = batch_size


	def fit(self,x_train,y_train):
		# initialising the coef
		self.intercept_ = 0
		self.coef_ = np.ones(x_train.shape[1])

		for i in range(self.epochs):

			for j in range(int(x_train.shape[0]/self.batch_size)):

				idx = np.random.choice(x_train.shape[0], self.batch_size,replace = False)
				y_hat = np.dot(x_train[idx],self.coef_) + self.intercept_

				intercept_der = -2 * np.mean(y_train[idx] - y_hat)
				self.intercept_ = self.intercept_ - (self.lr * intercept_der)

				coef_der = -2*np.dot((y_train[idx] - y_hat), x_train[idx])/len(idx)
				self.coef_ = self.coef_ - (self.lr * coef_der)

		print(self.intercept_, self.coef_)

	def predict(self,x_test):
		return np.dot(x_test,self.coef_) + self.intercept_


mbr = MBGDregressor(batch_size = int(x_train.shape[0]/10), learning_rate=0.01, epochs= 50)
# start = time.time()

mbr.fit(x_train,y_train)
# print("time taken is", time.time() - start)
print(mbr.predict(x_test))
y_pred = mbr.predict(x_test)
print(r2_score(y_test,y_pred))
