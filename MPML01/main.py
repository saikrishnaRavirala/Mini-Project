# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.sparse
from subprocess import check_output
dataset_url = 'Dataset.csv'
df = pd.read_csv(dataset_url,encoding = 'ISO-8859-1')
df.info()
df.dropna(how='any', inplace=True)
y_test=[]
quality=df["Rainfall"][(df["Quality Parameter"]==1)]
y_test.append(int(np.mean(quality.values)))
quality=df["Rainfall"][(df["Quality Parameter"]==2)]
y_test.append(int(np.mean(quality.values)))
quality=df["Rainfall"][(df["Quality Parameter"]==3)]
y_test.append(int(np.mean(quality.values)))
quality=df["Rainfall"][(df["Quality Parameter"]==4)]
y_test.append(int(np.mean(quality.values)))
quality=df["Rainfall"][(df["Quality Parameter"]==5)]
y_test.append(int(np.mean(quality.values)))
print(y_test)
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
df.groupby('State_Name').mean().sort_values(by='Rainfall', ascending=False)['Rainfall'].plot('bar', color='r',width=0.3,title='State wise Average Annual Rainfall', fontsize=20)
plt.xticks(rotation = 90)
plt.ylabel('Average Annual Rainfall (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
print(df.groupby('State_Name').mean().sort_values(by='Rainfall', ascending=False)['Rainfall'])
plt.show()
#Annual Graph
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)

dfg = df.groupby('Quality Parameter').sum()['Rainfall']
dfg.plot('line', title='Overall Rainfall in Each Year', fontsize=20)
#df.groupby('YEAR').sum()['Rainfall'].plot()
#plt.xlim(0, 115)
#plt.xticks(np.linspace(0,115,24,endpoint=True),np.linspace(1900,2015,24,endpoint=True).astype(int))
#plt.xticks(np.linspace(1901,2015,24,endpoint=True))
#plt.xticks(rotation = 90)
plt.ylabel('Overall Rainfall (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
print('Max: ' + str(dfg.max()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.max()].index.values[0:]))
print('Min: ' + str(dfg.min()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.min()].index.values[0:]))
print('Mean: ' + str(dfg.mean()))



fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
xlbls = ['Iron','Salinity','Fluoride','Arsenic','Nitrate'] 
#xlbls.sort()
dfg = df.groupby('Quality Parameter').mean()['Rainfall']
plt.xticks(np.linspace(0,35,36,endpoint=True),xlbls)
plt.xticks(  rotation = 90)
plt.ylabel('Rainfall (mm)')
plt.legend(loc='upper right', fontsize = 'xx-large')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
dfg=pd.DataFrame(dfg)
print(dfg)
dfg = dfg.mean(axis=1)
print('Max: ' + str(dfg.max())  )
print('Min: ' + str(dfg.min()))
print('Mean: ' + str(dfg.mean()))


df2 = df[['Quality Parameter','Rainfall']]


test_df=df[['State_Name','Quality Parameter','Rainfall']]

group_df=test_df.groupby(['State_Name','Rainfall'])
state=[]

for gd in group_df:   
    dg=pd.DataFrame(list(gd)[1])
    l=list(list(gd)[0])
    l.append(dg['Quality Parameter'].value_counts().idxmax())
    state.append(l)

state=pd.DataFrame(state)
state.columns=['State_Name','Rainfall','Quality Parameter']
state.to_csv('final.csv')
ss = df.groupby('Quality Parameter').mean()['Rainfall']

from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=1) 

X_train=df['Quality Parameter']
y_train=df.Rainfall
from sklearn.model_selection import train_test_split  
#X_train, X_test,y_train,y_test = train_test_split(x, y, test_size=0.3)  

X_test=[]
X_test.append([1])
X_test.append([2])
X_test.append([3])
X_test.append([4])
X_test.append([5])

X_train=X_train.values.reshape(X_train.shape[0],1)
y_train=y_train.values.reshape(y_train.shape[0],1)
#y_test=y_test.values.reshape(y_test.shape[0],1)
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)
X_train=X_train.astype('int')
y_train=y_train.astype('int')


classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test) 
print(X_test)
print(y_pred)
plt.ylabel('Water Quality')
plt.xlabel('Rainfall (mm)')
plt.scatter(y_train,X_train)
plt.plot(y_pred,X_test,color='red')
lab=['iron','salinity','fluoride','arsenic','nitrate'] 
Dic={'Water Quality':lab,'RainFall':y_pred}
plt.show();
d=pd.DataFrame(Dic);
d.to_csv('result.csv')

s='yes'
while True:
    if s=='yes':
        try:
            print("Please Enter RainFall....")
            r = int(input()) 
            quality=d["Water Quality"][(d["RainFall"]==min(y_pred, key=lambda x:abs(x-r)))]
            print("The Expected Water Quality is ",quality.values[0])
            print("Please Enter WaterQuality....")
            q = str(input()) 
            rain=d["RainFall"][(d["Water Quality"]==q)]
            print("The Expected AnnualRainfall is ",rain.values[0])
            print("Please Enter yes to continue")
            s = str(input())
        except:
            print("Enterd invalid Value")
            print("Please Enter yes to continue")
            s = str(input())
    else:
        print("You Entered %s to quit"%s)
        break;
    
