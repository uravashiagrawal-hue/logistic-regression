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

m= 76.96
class GDRegressor:
	def __init__(self,learning_rate,epochs):
		self.m = 76.96
		self.b = -120
		self.lr = learning_rate
		self.epochs = epochs

	def fit(self,x,y):
		# calculating the b using GD
		for i in range(self.epochs):
			loss_slope = -2*np.sum(y - self.m*x.ravel() - self.b)
			self.b = self.b - (self.lr - loss_slope)

		print(self.b)

gd= GDRegressor(0.01,10)
gd.fit(x,y)

