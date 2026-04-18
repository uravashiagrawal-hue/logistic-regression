import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
data = load_diabetes()
# to know about the dataset
print(data.DESCR)
x=data.data
y= data.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=45)

from sklearn.linear_model import LinearRegression
lr =LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

from sklearn.metrics import r2_score, mean_squared_error
print("r2 score", r2_score(y_test, y_pred))
print("rmse", np.sqrt(mean_squared_error(y_test, y_pred)))

# now with ridge
from sklearn.linear_model import Ridge
r= Ridge(alpha = 0.0001)
# here alpha is lembda as we learn in our theory
r.fit()
