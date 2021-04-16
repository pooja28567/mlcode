import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
food_data = pd.read_csv('project1.csv')
x=food_data.drop('Target',axis=1)
y=food_data['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
a = [[32,3,323]]
target1 = logmodel.predict(a)
if target1=='0':
    print('Food Sample is not spoilt')
else:
    print('Food Sample is spoilt')