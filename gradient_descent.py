# first we apply OLS(ordinary least square)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('placement.csv')
print(df.head())

plt.scatter(df['cgpa'], df['package'])
plt.xlabel('CGPA')
plt.ylabel('package(in lpa)')
plt.show()

# sabse phele x and y ko alag alag karte hai by
x = df.iloc[:, 0].values.reshape(-1,1)
y = df.iloc[:,-1]
print(x)
print(y)

from sklearn.model_selection import train_test_split
# dividing our data into 4 parts
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_train,y_train)
print(lr.coef_)
print(lr.intercept_)

plt.scatter(x,y)
plt.plot(x,lr.predict(x), color = 'red')
plt.show()

# lets apply gredient descent assuming slope is constant m=0.55795
# and let's assume the starting valie for intercept b=0
y_pred = ((0.55795 * x) +0)

plt.scatter(x,y)
plt.plot(x,lr.predict(x), color = 'red',label = 'OLS')
plt.plot(x,y_pred,color = 'green', label = 'b=o')
plt.legend()
plt.show()

m = 0.55795
b=0
loss_slope = -2 * np.sum(y- m*x.ravel()  -b)
print(loss_slope)

learning_rate = 0.0001
step_size = loss_slope * learning_rate
print(step_size)

# calculating new intercept
b= b-step_size
print(b)

# again
y_pred1 = ((0.55795 * x) +b)

plt.scatter(x,y)
plt.plot(x,lr.predict(x), color = 'red',label = 'OLS')
plt.plot(x,y_pred,color = 'green', label = 'b={}'.format(b))
plt.plot(x,y_pred,color = 'blue', label = 'b=0')
plt.legend()
plt.show()

loss_slope = -2 * np.sum(y- m*x.ravel()  -b)
print(loss_slope)

step_size = loss_slope * learning_rate
print(step_size)

b= b-step_size
print(b)

# again
y_pred2 = ((0.55795 * x) +b)

plt.scatter(x,y)
plt.plot(x,lr.predict(x), color = 'red',label = 'OLS')
plt.plot(x,y_pred,color = 'yellow', label = 'b={}'.format(b))
plt.plot(x,y_pred,color = 'green', label = 'b={}'.format(b))
plt.plot(x,y_pred,color = 'blue', label = 'b=0')
plt.legend()
plt.show()
