import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_regression
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# making a dataset
x,y = make_regression(n_samples=100, n_features=2,n_targets=1,n_informative=2, noise = 50)
df = pd.DataFrame({'feature1':x[:,0],'feature2':x[:,1],'target':y})
print(df.shape)

print(df.head())
fig = px.scatter_3d(df,x='feature1', y='feature2',z='target')
fig.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=3)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

print('MAE', mean_absolute_error(y_test,y_pred))
print('MSE',mean_squared_error(y_test,y_pred))
print('r2-score', r2_score(y_test,y_pred))
