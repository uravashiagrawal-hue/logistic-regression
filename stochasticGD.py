from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time

x,y = load_diabetes(return_X_y=True)
print(x.shape)
print(y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=2)

from sklearn.linear_model import SGDRegressor
reg =SGDRegressor(max_iter=45, learning_rate='constant',eta0 = 0.01)

reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print(r2_score(y_test, y_pred))
