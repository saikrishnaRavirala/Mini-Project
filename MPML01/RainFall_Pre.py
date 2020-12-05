# -*- coding: utf-8 -*-
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats
#Water Quality Pre_Proccessing

df=pd.read_csv(r'rainfall.csv',encoding='ISO-8859-1')
print(df.head())
print(df.shape)
with open('rainfall.csv', 'r') as inp, open('rainfall_edit.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if not 'NA' in set(row):
            writer.writerow(row)
        else:
            print(row)
            

    
df=pd.read_csv('rainfall_edit.csv',encoding='ISO-8859-1')
print(df.head())
print(df.shape)   
df.hist()
fig = plt.gcf()
fig.set_size_inches(15,20,forward=True)
fig.savefig('test1png.png', dpi=100)
plt.show()

df.plot(kind='box',subplots=True,sharex=False,sharey=False)
fig = plt.gcf()
fig.set_size_inches(15,20,forward=True)
fig.savefig('test2png.png', dpi=100)
plt.show()

with open('rainfall_edit.csv','r') as in_file, open('rainfall_e.csv','w') as out_file:
    NEWSET = set()
    for line in in_file:
        if not line in NEWSET:
            NEWSET.add(line)
            out_file.write(line)
         


data1 = pd.read_csv(r'rainfall_e.csv',encoding='ISO-8859-1', error_bad_lines=False)

print(data1)

X_train1, X_test1= train_test_split(data1,test_size=0.3,random_state=100)
print("\nX_train1:\n")
print(X_train1.head())
print(X_train1.shape)

print("\nX_test1:\n")
print(X_test1.head())
print(X_test1.shape)



