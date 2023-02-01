#Assignment on Naive Bayes Classifier

import pandas as pd
import numpy as np
data=pd.read_csv("iris.csv")
x=data.iloc[:,:-1].values
#print(x)
y=data.iloc[:,-1].values
#print(y)
#Encoding the category
from sklearn.preprocessing import LabelEncoder
y=LabelEncoder().fit_transform(y)
#print(y)
#split train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
#print(x_train)
#print(y_train)
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB(fit_prior=False)
mnb.fit(x_train,y_train)
#Prediction using Bayes
y_predict=mnb.predict(x_test)
print(y_predict)
print(y_test)
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test,y_predict)
print(confusion_mat)
#Classifier Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)*100
print(accuracy)