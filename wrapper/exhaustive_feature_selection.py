from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import pandas as pd

df = pd.read_csv('wrapper\iris.csv')
print(df.head())

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

lr = LogisticRegression()
sel = EFS(lr, max_features=4, cv=5)

model = sel.fit(df.iloc[:,:4], df['Species'])
print(model.best_score_)

print(model.best_feature_names_)
print(model.subsets_)
# converting subset data from dict to dataframe
metric_df = pd.DataFrame.from_dict(model.get_metric_dict()).T
print(metric_df)

import matplotlib.pyplot as plt
plt.plot([str(k) for k in metric_df['feature_names']], metric_df['avg_score'])
# plt.plot(metric_df['feature_names'], metric_df['avg_score'])
plt.xticks(rotation=90)
plt.show()
