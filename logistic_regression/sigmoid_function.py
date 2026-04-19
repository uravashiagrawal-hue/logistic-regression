import numpy as np
from sklearn.datasets import make_classification
x,y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=30)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.scatter(x[:,0],x[:,1], c=y, cmap ='winter', s=100)
plt.show()


# step function
def perceptron(x,y):
	x = np.insert(x,0,1,axis =1)
	weights = np.ones(x.shape[1])
	lr = 0.1

	for i in range(1000):
		j = np.random.randint(0,100)
		y_hat = step(np.dot(x[j], weights))
		weights = weights + lr*(y[j]-y_hat)*x[j]

	return weights[0], weights[1:]

def step(z):
	return 1 if z>0 else 0

intercept_,coef_ = perceptron(x,y)
print(coef_)
m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])
x_input = np.linspace(-3,3,100)
y_input = m*x_input + b
plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.scatter(x[:,0],x[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)
plt.show()

# through logistic regression
from sklearn.linear_model import LogisticRegression
lor = LogisticRegression()
lor.fit(x,y)

m = -(lor.coef_[0][0]/lor.coef_[0][1])
b = -(lor.intercept_/lor.coef_[0][1])
x_input1 = np.linspace(-3,3,100)
y_input1 = m*x_input1 + b

plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.plot(x_input1,y_input1,color='black',linewidth=3)
plt.scatter(x[:,0],x[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)
plt.show()




# by using sigmoid function
def perceptron(X,y):

    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1

    for i in range(1000):
        j = np.random.randint(0,100)
        y_hat = sigmoid(np.dot(X[j],weights))
        weights = weights + lr*(y[j]-y_hat)*X[j]

    return weights[0],weights[1:]

def sigmoid(z):
    return 1/(1 + np.exp(-z))
