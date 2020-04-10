import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

gene_means = []
gene_stds  = []

all_chromosomes = []

for i in range(10):
    filepath = "data/gen{}.npy".format(str(i))
    chromosomes = np.load(filepath)
    print(filepath)
    #print(chromosomes)
    means   = np.mean(chromosomes, axis=0)
    std     = np.std(chromosomes,  axis=0)
    gene_means.append(means)
    gene_stds.append(std)

    all_chromosomes.append(chromosomes)

gens = range(10)

gene_means = np.array(gene_means)
gene_stds  = np.array(gene_stds)

plt.plot(gens, gene_means[:,0])
plt.fill_between(gens, gene_means[:,0] - gene_stds[:,0], gene_means[:,0] + gene_stds[:,0], color="#7ff58f", alpha=0.6)
plt.show()



print(all_chromosomes[0])
incrementer = 255 / len(all_chromosomes)
index = 0
for generation in all_chromosomes:
    print(generation[:,0])
    
    color = cm.viridis(index)
    index += int(incrementer)
    plt.scatter(generation[:,0], generation[:,1], c=color)
    
plt.show()
    