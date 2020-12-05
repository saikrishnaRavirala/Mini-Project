# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from pandas.plotting import scatter_matrix
data = pd.read_csv("Dataset.csv",encoding='latin1')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
# scatter plot matrix
scatter_matrix(data)
plt.show()
X = data.iloc[:, 7:8].values 
y = data.iloc[:, 8].values 
print(y)
# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
lin = LinearRegression() 

lin.fit(X, y) 
# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 

poly = PolynomialFeatures(degree = 3) 
X_poly = poly.fit_transform(X) 

poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 
# Visualising the Linear Regression results 

State_Data = data[['State_Name', 'Quality Parameter']]
from sklearn.preprocessing import LabelEncoder
numbers = LabelEncoder()
State_Quality_Count = pd.DataFrame({'count' : State_Data.groupby( [ "State_Name","Quality Parameter"] ).size()}).reset_index()
plt.figure(figsize=(6,4))
x = State_Quality_Count.groupby('State_Name')
plt.rcParams['figure.figsize'] = (9.5, 6.0)
genre_count = sns.barplot(y='Quality Parameter', x='count', data=State_Quality_Count, ci=None)
plt.show()

plt.scatter(X, y, color = 'blue') 

plt.plot(X, lin.predict(X), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('Year') 
plt.ylabel('RainFall') 

plt.show() 
# Visualising the Polynomial Regression results 
plt.scatter(X, y, color = 'blue') 
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('Year') 
plt.ylabel('RainFall') 
plt.show() 
# Predicting a new result with Linear Regression 
print(lin.predict(X+1))
# Predicting a new result with Polynomial Regression 
print(lin2.predict(poly.fit_transform(X+1)))

