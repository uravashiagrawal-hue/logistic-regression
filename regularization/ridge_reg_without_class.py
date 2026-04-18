from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import numpy as np

x,y = load_diabetes(return_X_y=True)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=4)
from sklearn.linear_model import Ridge
reg = Ridge(alpha = 0.1, solver='cholesky')
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
print(r2_score(y_test,y_pred))
print(reg.coef_)
print(reg.intercept_)
# this is with the sklearn inbuilt class

