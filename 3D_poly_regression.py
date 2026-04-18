import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

x = 7* np.random.rand(100,1) -2.8
y= 7* np.random.rand(100,1) - 2.8

z= x**2 + y**2 + 0.2*x + 0.2*y + 0.1*x*y +2 + np.random.randn(100,1)
# z =x^2 + y^2 + 0.2x + 0.2y + 0.1xy + 2

import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(df, x=x.ravel(), y= y.ravel(), z=z.ravel())
fig.show()
