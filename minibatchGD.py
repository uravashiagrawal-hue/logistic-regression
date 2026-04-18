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

from sklearn.linear_model import SGDRegressor
reg =SGDRegressor(learning_rate='constant',eta0 = 0.2)
batch_size = 35

for i in range(100):
	idx = random.choice(x_train.shape[0],batch_size, replace = False)
	reg.partial_fit(x_train[idx], y_train[idx])

print(reg.coef_)
print(reg.intercept_)
y_pred = reg.predict(x_test)
print(r2_score(y_test, y_pred))
