import numpy as np

x = np.loadtxt("QSimple/winrates_movetype_2.txt") / 10
print(x)

print(np.mean(x))
print(np.std(x))