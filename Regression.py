#Assignment based on simple linear regression on any dataset
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("houseprices.csv")
print(data)
x=data.iloc[0:100,1:2].values
print(x)
y=data.iloc[0:100,:1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
print(x_train)
print(len(x_train))
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
predicted=reg.predict(x_test)
print("intercept \n",reg.intercept_)
print("slope or weight \n",reg.coef_)
from sklearn.metrics import mean_squared_error
print("mean squared error is: \n",mean_squared_error(y_test,predicted))
print("predicted data by model \n",predicted)
print("Actual data \n",y_test)
plt.scatter(x_test,y_test)
plt.xlabel("area")
plt.ylabel("price")
plt.title("houseprices using linear regression \n")
plt.plot(x_test,predicted)
plt.show()
print("price: \n",reg.predict([[1802]]))
