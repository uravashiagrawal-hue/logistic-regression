import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
print(df.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.iloc[:,:-1], df['medv'], test_size=0.2, random_state=1)
print(x_train.shape)
print(x_test.shape)

print(x_train.head())

# when the dtd is in different scale- frist data ko ek scale pr leke aao
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# baseline model - phele model ko complete data pr train kr rhe hai
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
model = LinearRegression()

print("training", np.mean(cross_val_score(model,x_train, y_train,cv=5,scoring = 'r2')))
print("testing", np.mean(cross_val_score(model,x_test, y_test,cv=5,scoring = 'r2')))


# now appling EFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

lr = LinearRegression()
exh = EFS(lr, max_features=13, scoring='r2', cv = 10, print_progress = True, n_jobs=-1)
sel = exh.fit(x_train, y_train)
print(sel.best_score_)
print(sel.best_feature_names_)

metric_df = pd.DataFrame.from_dict(sel.get_metric_dict()).T
print(metric_df)
