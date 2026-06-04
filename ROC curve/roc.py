import pandas as pd
data = pd.read_csv('ROC curve\diabetes (1).csv')
print(data.head())
print(data.columns)
x= data.drop('Outcome', axis=1)
y= data['Outcome']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_score = model.predict_proba(x_test)[:,1]
print(y_score)

from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(y_test, y_score)
print(threshold)
