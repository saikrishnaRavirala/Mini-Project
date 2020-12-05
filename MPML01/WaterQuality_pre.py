import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats
#Water Quality Pre_Proccessing

df=pd.read_csv(r'WaterQuality.csv',encoding='ISO-8859-1')
print(df.head())
print(df.shape)
with open('WaterQuality.csv', 'r') as inp, open('WaterQuality_edit.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if not 'NA' in set(row):
            writer.writerow(row)
        else:
            print(row)
            

df=pd.read_csv('WaterQuality_edit.csv',encoding='ISO-8859-1')
print(df.head())
print(df.shape)
df.hist()
plt.show()

df.plot(kind='box',subplots=True,sharex=False,sharey=False)
plt.show()

with open('WaterQuality_edit.csv','r') as in_file, open('WaterQuality.csv','w') as out_file:
    NEWSET = set()
    for line in in_file:
        if not line in NEWSET:
            NEWSET.add(line)
            out_file.write(line)
'''            
z=np.abs(stats.zscore(df))
print("z value is",z)
print("df",df.shap)
df_o=df[(z<3).all(axis=1)]
print("df_o",df_o.shap)'''
data1 = pd.read_csv(r'WaterQuality.csv',encoding='ISO-8859-1', error_bad_lines=False)
print(data1.groupby("Block Name").size())
Block_Data = data1[['Block Name', 'Quality Parameter']]
print(data1)
print(Block_Data)

X_train1, X_test1= train_test_split(data1,test_size=0.3,random_state=100)
print("\nX_train1:\n")
print(X_train1.head())
print(X_train1.shape)
print("\nX_test1:\n")
print(X_test1.head())
print(X_test1.shape)


