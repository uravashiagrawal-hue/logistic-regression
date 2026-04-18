import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_regression

x,y = make_regression(n_samples = 100, n_features = 1,n_informative=1, n_targets=1, noise = 20)
plt.scatter(x,y)
plt.show()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x,y)
print(lr.coef_)
print(lr.intercept_)
print(y.shape)

from sklearn.model_selection import cross_val_score
print(np.mean(cross_val_score(lr,x,y,cv = 10,scoring='r2')))

m= 76.96
class GDRegressor:
	def __init__(self,learning_rate,epochs):
		self.m = 1
		self.b = 0
		self.lr = learning_rate
		self.epochs = epochs

	def fit(self,x,y):
		n=len(y)
		# calculating the b using GD
		for i in range(self.epochs):
			loss_slope_b = -2/n * np.sum(y - self.m*x.ravel() - self.b)
			loss_slope_m = -2/n * np.sum(y - self.m*x.ravel() - self.b)*x.ravel()
			self.b = self.b - (self.lr * loss_slope_b)
			self.m = self.m - (self.lr * loss_slope_m)


		print(self.m,self.b)

	def predict(self,x):
		return self.m * x + self.b

gd= GDRegressor(0.001,100)
gd.fit(x,y)
gd.predict(x)
