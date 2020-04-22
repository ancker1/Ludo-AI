import matplotlib.pyplot as plt
import numpy as np
import sg_filter
#generations = np.array(list(range(32))) * 10
#winrate = np.array([674, 785, 805, 793, 821, 841, 873, 874, 866, 894, 893, 875, 896, 900, 876, 909, 912, 903, 908, 914, 915, 917, 920, 897, 872, 902, 922, 916, 920, 890, 912, 905])

#plt.plot(generations, winrate/1000)
#plt.xlabel('Generation')
#plt.ylabel('Winrate')
#plt.show()

#x = np.array([904, 899, 901, 900, 920, 904, 905, 891, 904, 909, 904, 892, 899, 898, 903, 910, 917, 914, 912, 916, 902, 904, 901, 874, 907, 920, 896, 906, 915, 897])

#np.savetxt("analyze/test.txt",x)

x = np.loadtxt("../Evaluation/GASimple_vs_QLearner.txt") / 1000
#x = np.loadtxt("GASimple/gen_vs_random.txt") / 1000
print(x)

print(np.mean(x))
print(np.std(x))


y = np.loadtxt("GASimple/gen_vs_random.txt") / 10
gens = range(len(y))

fig1 = plt.figure(1)
plt.plot(gens, y, alpha=.3)
plt.plot(sg_filter.savitzky_golay(np.array(y), 15, 3))
plt.grid()
#plt.legend(['Win rate','Running mean(15)'])
plt.xlabel('Generation', fontsize=16)
plt.ylabel('Win rate [%]', fontsize=16)
plt.show()