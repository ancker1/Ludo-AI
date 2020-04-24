import matplotlib.pyplot as plt
import numpy as np
import argparse
import sg_filter

parser = argparse.ArgumentParser()
parser.add_argument('--individual',     type=int,   default = 0     )
args = parser.parse_args()

if args.individual == 0:
    generation_max = 101
    pathprefix = "GASimple"
    incrementer = 1
else:
    generation_max = 312
    pathprefix = "GANN"
    incrementer = 1

current_generation = 0
x = []
y = []

xmean = []
ymean = []
while current_generation < generation_max:
    winrates = np.loadtxt(pathprefix+"/generational_winrates/gen{}.txt".format(str(current_generation))) / 10
    xmean.append(current_generation)
    ymean.append(np.mean(winrates))
    for j in range(len(winrates)):
        x.append(current_generation)
        y.append(winrates[j])
    if current_generation == 20:
        incrementer = 5 # analyze all up to gen 20 - hereafter only every fifth.
    current_generation += incrementer

plt.figure(figsize=(5,4))
plt.scatter(x, y, alpha = 0.1)
plt.plot(xmean,sg_filter.savitzky_golay(np.array(ymean), 15, 3), linewidth=3)
plt.xlabel('Generation')
plt.ylabel('Win rate [%]')
plt.show()

#scores = np.array(scores)
#np.savetxt(pathprefix+"/gen_vs_random.txt", scores)



