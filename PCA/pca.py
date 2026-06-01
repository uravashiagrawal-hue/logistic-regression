import numpy as np
import pandas as pd

# creating dataframe
np.random.seed(23)
mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1,20)

df = pd.DataFrame(class1_sample, columns = ['feature1','feature2','feature3'])
df['target'] = 1

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2,20)
df1 = pd.DataFrame(class2_sample, columns = ['feature1','feature2','feature3'])
df1['target'] = 0

df = pd.concat([df,df1], ignore_index = True)
df = df.sample(40)
print(df.head())

import plotly.express as px
fig = px.scatter_3d(df, x= df['feature1'], y = df['feature2'], z= df['feature3'], color=df['target'].astype('str'))
fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig.show()

# step 1 - apply standard scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df.iloc[:,0:3] = scaler.fit_transform(df.iloc[:,0:3])


# step 2 to find covariance matrix
covariance_matrix = np.cov([df.iloc[:,0],df.iloc[:,1],df.iloc[:,2]])
print('Covariance Matrix:\n', covariance_matrix)

# finding EV and EVs
eigen_values, eigen_vectors= np.linalg.eig(covariance_matrix)
print(eigen_values)
print(eigen_vectors)


