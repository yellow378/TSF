import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
data_path = "../wtbdata_245days.csv"

df = pd.read_csv(data_path)
data = df[['Wspd','Patv']][0:1440]
data.to_csv("10days.csv")
'''
data_path = "./10days.csv"
df = pd.read_csv(data_path)
first_day = df[0:144]
second_day = df[144:144+144]

plt.figure(figsize=(10, 6))
fig, ax1 = plt.subplots()
l1 = ax1.plot(first_day[['Wspd']].to_numpy().reshape(-1), label='wind speed', color='blue')
ax1.set_xlabel('Time')
ax1.set_ylim([-1,20])
ax1.set_ylabel('wind speed',color='b')
ax1.tick_params(axis='y',labelcolor='b')

ax2 = plt.twinx()
l2 = ax2.plot(first_day[['Patv']].to_numpy().reshape(-1), label='wind power', color='red')
ax2.set_ylabel('wind power',color='r')
ax2.set_ylim([-1,2000])
ax2.tick_params(axis='y',labelcolor='r')

fig2, ax21 = plt.subplots()
l21 = ax21.plot(second_day[['Wspd']].to_numpy().reshape(-1), label='wind speed', color='blue')
ax21.set_xlabel('Time')
ax21.set_ylim([-1,20])
ax21.set_ylabel('wind speed',color='b')
ax21.tick_params(axis='y',labelcolor='b')

ax22 = plt.twinx()
l22 = ax22.plot(second_day[['Patv']].to_numpy().reshape(-1), label='wind power', color='red')
ax22.set_ylabel('wind power',color='r')
ax22.set_ylim([-1,2000])
ax22.tick_params(axis='y',labelcolor='r')
plt.show()
