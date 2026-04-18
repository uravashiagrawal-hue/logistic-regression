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
x= df.iloc[:,0:1]
y = df.iloc[:,-1]
print(x)
print(y)

from sklearn.model_selection import train_test_split
# dividing our data into 4 parts
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_train,y_train)
print(x_test)
print(y_test)

print(lr.predict(x_test.iloc[0].values.reshape(1,1)))

# how the linear regression form line in the give data plot
plt.scatter(df['cgpa'], df['package'])
plt.plot(x_train, lr.predict(x_train), color='red')
plt.xlabel('CGPA')
plt.ylabel('package(in lpa)')
plt.show()


# error

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
y_prediction = lr.predict(x_test)
print(y_prediction)
print('MAE :', mean_absolute_error(y_test, y_prediction))
print('MSE :', mean_squared_error(y_test, y_prediction))
print('RMSE :', np.sqrt(mean_squared_error(y_test, y_prediction)))
print('R2 SCORE:', r2_score(y_test,y_prediction))
r2 = r2_score(y_test,y_prediction)
