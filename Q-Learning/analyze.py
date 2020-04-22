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
            trainingRewards.append(float(row[0]))

#plt.plot(trainingRewards, alpha=.2, label = "orig")
plt.plot(np.convolve(trainingRewards, np.ones((10,))/10, mode='valid'), alpha=.2, label = "10")
#plt.plot(np.convolve(trainingRewards, np.ones((100,))/100, mode='valid'), label = "100")
#plt.plot(np.convolve(trainingRewards, np.ones((300,))/300, mode='valid'), label = "300")
plt.plot(sg_filter.savitzky_golay(np.array(trainingRewards), 301, 3))
plt.xlabel('Episodes', fontsize=16)
plt.ylabel('Accumulated reward', fontsize=16)
plt.grid()
plt.show()