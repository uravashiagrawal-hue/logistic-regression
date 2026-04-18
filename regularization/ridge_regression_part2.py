from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

x,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise =20, random_state=13)
plt.scatter(x,y)
plt.show()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)
print(lr.coef_)
print(lr.intercept_)

from sklearn.linear_model import Ridge
rr = Ridge(alpha=10)
rr.fit(x,y)
print(rr.coef_)
print(rr.intercept_)


rr1 = Ridge(alpha =100)
rr1.fit(x,y)
print(rr1.coef_)
print(rr1.intercept_)

plt.plot(x,y,'b.')
plt.plot(x,lr.predict(x), color = 'red', label = 'aplha=0')
plt.plot(x,rr.predict(x), color = 'green', label ='aplha = 10')
plt.plot(x,rr1.predict(x), color ='orange', label = 'alpha =100')
plt.legend()
plt.show()
