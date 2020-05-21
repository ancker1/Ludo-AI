import matplotlib.pyplot as plt
from os import path
import numpy as np
import csv
import sg_filter

trainingRewards = []
if path.isfile("rewards.csv"):
    rewards = csv.reader(open("rewards.csv"))
    for row in rewards:
        #print(row)
        if row:
            trainingRewards.append(float(row[0])/100)
plt.figure(figsize=(5.5,4))
#plt.plot(trainingRewards, alpha=.2, label = "orig")
plt.plot(np.convolve(trainingRewards, np.ones((10,))/10, mode='valid'), alpha=.2, label = "10")
#plt.plot(np.convolve(trainingRewards, np.ones((100,))/100, mode='valid'), label = "100")
#plt.plot(np.convolve(trainingRewards, np.ones((300,))/300, mode='valid'), label = "300")
plt.plot(sg_filter.savitzky_golay(np.array(trainingRewards), 301, 3))
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Accumulated reward', fontsize=12)
plt.grid()
plt.xticks(ticks=[0, 10000, 20000, 30000, 40000])

plt.figure(figsize=(5.5,4))
rewards = np.loadtxt("QSimple/training/rewards.txt")
print(rewards)
plt.plot(rewards, alpha=.2)
plt.plot(sg_filter.savitzky_golay(rewards, 31, 3))
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Accumulated reward', fontsize=12)
plt.grid()
plt.show()