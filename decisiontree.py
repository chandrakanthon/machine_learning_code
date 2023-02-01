#Assignment on Decision Tree Algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("weather.csv")
print(data.columns  )

from sklearn.preprocessing import LabelEncoder
data['outlook']=LabelEncoder().fit_transform(data['outlook'])
data['temperature']=LabelEncoder().fit_transform(data['temperature'])
data['humidity']=LabelEncoder().fit_transform(data['humidity'])
data['windy']=LabelEncoder().fit_transform(data['windy'])
data['play']=LabelEncoder().fit_transform(data['play'])
print(data)
# Lets split the training data and its coresponding prediction values.
# y - holds all the decisions.
# X - holds the training data.
y= data['play']
x = data.drop(['play'],axis=1)
# Fitting the model
from sklearn import tree
dt=tree.DecisionTreeClassifier(criterion="entropy")
dt=dt.fit(x,y)
tree.plot_tree(dt)
plt.show()