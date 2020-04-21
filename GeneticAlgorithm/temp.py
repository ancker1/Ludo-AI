import matplotlib.pyplot as plt
import numpy as np
#generations = np.array(list(range(32))) * 10
#winrate = np.array([674, 785, 805, 793, 821, 841, 873, 874, 866, 894, 893, 875, 896, 900, 876, 909, 912, 903, 908, 914, 915, 917, 920, 897, 872, 902, 922, 916, 920, 890, 912, 905])

#plt.plot(generations, winrate/1000)
#plt.xlabel('Generation')
#plt.ylabel('Winrate')
#plt.show()

#x = np.array([904, 899, 901, 900, 920, 904, 905, 891, 904, 909, 904, 892, 899, 898, 903, 910, 917, 914, 912, 916, 902, 904, 901, 874, 907, 920, 896, 906, 915, 897])

#np.savetxt("analyze/test.txt",x)

x = np.loadtxt("analyze/GASimple_vs_random_adjust.txt") / 1000
print(x)

print(np.mean(x))
print(np.std(x))