import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def analyzer(max_depth):
	data = pd.read_csv('Decision tree\Social_Network_Ads.csv')
	x=data.iloc[:,2:4].values
	y= data.iloc[:,-1].values

	clf = DecisionTreeClassifier(max_depth=max_depth)
	clf.fit(x,y)

	a= np.arange(start=x[:,0].min()-1, stop=x[:,0].max()+1, step = 0.1)
	b=np.arange(start=x[:,1].min()-1, stop=x[:,1].max()+1, step = 100)
	XX,YY = np.meshgrid(a,b)
	input_array = np.array([XX.ravel(), YY.ravel()]).T
	labels = clf.predict(input_array)
	plt.contourf(XX,YY, labels.reshape(XX.shape),alpha = 0.5)
	plt.scatter(x[:,0],x[:,1], c=y)
	plt.show()

analyzer(max_depth=3)
