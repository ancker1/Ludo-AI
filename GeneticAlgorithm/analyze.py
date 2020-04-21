import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

gene_means = []
gene_stds  = []

all_chromosomes = []

N_GENERATIONS = 100

for i in range(N_GENERATIONS):
    filepath = "data/gen{}.npy".format(str(i))
    chromosomes = np.load(filepath)
    #print(filepath)
    #print(chromosomes)
    means   = np.mean(chromosomes, axis=0)
    std     = np.std(chromosomes,  axis=0)
    gene_means.append(means)
    gene_stds.append(std)

    all_chromosomes.append(chromosomes)

gens = range(N_GENERATIONS)

gene_means = np.array(gene_means)
gene_stds  = np.array(gene_stds)

#fig1 = plt.figure(1)
#plt.plot(gens, gene_means[:,0])
#plt.fill_between(gens, gene_means[:,0] - gene_stds[:,0], gene_means[:,0] + gene_stds[:,0], color="#7ff58f", alpha=0.6)
#plt.show()



fig2, axs = plt.subplots(2,2)
axs[0,0].plot(gens, gene_means[:,0])
axs[0,0].fill_between(gens, gene_means[:,0] - gene_stds[:,0], gene_means[:,0] + gene_stds[:,0], color="#7ff58f", alpha=0.6)
#axs[0,0].xlabel('Generation')
axs[0,0].set_ylabel(r'$\omega_0$')

axs[0,1].plot(gens, gene_means[:,1])
axs[0,1].fill_between(gens, gene_means[:,1] - gene_stds[:,1], gene_means[:,1] + gene_stds[:,1], color="#7ff58f", alpha=0.6)
#axs[0,1].xlabel('Generation')
axs[0,1].set_ylabel(r'$\omega_1$')

axs[1,0].plot(gens, gene_means[:,2])
axs[1,0].fill_between(gens, gene_means[:,2] - gene_stds[:,2], gene_means[:,2] + gene_stds[:,2], color="#7ff58f", alpha=0.6)
axs[1,0].set_xlabel('Generation')
axs[1,0].set_ylabel(r'$\omega_2$')

axs[1,1].plot(gens, gene_means[:,3])
axs[1,1].fill_between(gens, gene_means[:,3] - gene_stds[:,3], gene_means[:,3] + gene_stds[:,3], color="#7ff58f", alpha=0.6)
axs[1,1].set_xlabel('Generation')
axs[1,1].set_ylabel(r'$\omega_3$')

#print(all_chromosomes[0])
fig2 = plt.figure(2)
incrementer = 1 / len(all_chromosomes)
index = 0
for generation in all_chromosomes:
    #print(generation[:,0])
    color = cm.viridis(index)
    index += incrementer
    plt.scatter(generation[:,0], generation[:,1], c=color, s = 1)
cbar = plt.colorbar()
cbar.ax.get_yaxis().set_ticks([])
for i in range(10):
    cbar.ax.text(1.5, (i + 1)*.098, str(int(N_GENERATIONS* ((i + 1)*.1))))
plt.xlabel(r'$\omega_0$')
plt.ylabel(r'$\omega_1$')

fig3 = plt.figure(3)
incrementer = 1 / len(all_chromosomes)
index = 0
for generation in all_chromosomes:
    #print(generation[:,0])
    color = cm.viridis(index)
    index += incrementer
    plt.scatter(generation[:,2], generation[:,3], c=color, s = 1)
    print(index)
cbar = plt.colorbar()
cbar.ax.get_yaxis().set_ticks([])
for i in range(10):
    cbar.ax.text(1.5, (i + 1)*.098, str(int(N_GENERATIONS* ((i + 1)*.1))))
plt.xlabel(r'$\omega_2$')
plt.ylabel(r'$\omega_3$')

plt.show()
    