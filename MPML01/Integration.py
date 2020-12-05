import warnings 
import pandas as pd
import csv
import sklearn
from sklearn.preprocessing import LabelEncoder
numbers = LabelEncoder()
wq = pd.read_csv('WaterQuality.csv',encoding='latin1')
rf = pd.read_csv('RainFall.csv')
'''
rf['Quality'] = numbers.fit_transform(rf['Quality Parameter'])
State_Quality_Count = pd.DataFrame({'count' : rf.groupby( [ "State Name", "Year"] ).size()}).reset_index()
print(State_Quality_Count['State Name'],State_Quality_Count['Year'])
'''

with open(r'WaterQuality.csv',encoding='latin1') as inp, open('Dataset.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        with open('RainFall.csv','r') as inp:
            for clm in csv.reader(inp):
                if clm[0]==row[0] and clm[1]==row[7]:
                    row.append(clm[14])
                    print(row)
                    writer.writerow(row)