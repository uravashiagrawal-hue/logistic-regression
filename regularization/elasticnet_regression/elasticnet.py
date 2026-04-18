from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x,y = load_diabetes(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=2)

# linear regression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(r2_score(y_test, y_pred))

# ridge
reg1 = Ridge(alpha=0.1)
reg1.fit(x_train,y_train)
y_pred1 = reg1.predict(x_test)
print(r2_score(y_test, y_pred1))

# lasso
reg2 = Lasso(alpha = 0.01)
reg2.fit(x_train,y_train)
y_pred2 = reg2.predict(x_test)
print(r2_score(y_test, y_pred2))
