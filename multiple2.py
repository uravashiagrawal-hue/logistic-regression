import numpy as np
from sklearn.datasets import load_diabetes

x,y = load_diabetes(return_X_y=True)
print(x)
print(x.shape)
# shape is (422,10) - we have data of 422 patients and input values are 10 so we need B0,B1,B2,B3....B10
print(y)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.2,random_state=3)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

print(reg.coef_)   #reg.coef = B1,B2,B3....,B10

print(reg.intercept_)    #reg.intercept = B0


