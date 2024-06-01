import pandas as pd
import numpy as np
import os
import re

def timeProcess(x):
    time = x['Tmstamp']
    searchObj = re.search(r'([0-9]*):([0-9]*)',time)
    h = int(searchObj.group(1))
    m = int(searchObj.group(2))
    time = h*60+m
    x['Tmstamp'] = time
    return x
if not os.path.exists('processed'):
    os.mkdir('processed')
df = pd.read_csv('wtbdata_245days.csv')
df = df.fillna(0)
print(df.columns) 
for i in range(1,134):
    print(i)
    df1 = df[(df['TurbID']==i)]
    #df1 = df1.drop(['TurbID'],axis=1)
    # df1 = df1.apply(
    #     lambda x : timeProcess(x)
    #     ,axis=1
    # )
    df1 = df1[['Wspd','Patv']]
    df1.to_csv(f'processed/Turb{i}.csv',index=False)
    