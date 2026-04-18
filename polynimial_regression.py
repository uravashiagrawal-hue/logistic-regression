import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# generating the data
x= 6*np.random.rand(200,1) -3
y= 0.8 * x**2 + 0.9 * x + 2 + np.random.rand(200,1)
# y= 0.8x^2 + 0.9x +2

plt.plot(x,y,'b.')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state =2)
# applying polynomial linear regression
# degree 2
# creating object of class ploynomialfeatures under sklearn
poly = PolynomialFeatures(degree = 2)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.transform(x_test)

print(x_train[0])
print(x_train_trans[0])

lr = LinearRegression()
lr.fit(x_train_trans,y_train)

y_pred = lr.predict(x_test_trans)
print(r2_score(y_test, y_pred))
print(lr.coef_)
print(lr.intercept_)

