import numpy as np;
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

x,y = make_regression(n_samples =100, n_features=1, n_informative=1, n_targets=1,noise =20, random_state=13)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
plt.scatter(x,y)
plt.show()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.coef_)
print(lr.intercept_)

alphas = [0,1,5,10,30]
plt.figure(figsize=(12,6))
plt.scatter(x,y)
for i in alphas:
	L =Lasso(alpha =i)
	L.fit(x_train, y_train)
	plt.plot(x_test,L.predict(x_test),label = 'alpha={}'.format(i))
plt.legend()
plt.show()
