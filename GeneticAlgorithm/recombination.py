from individual import GAIndividual
import numpy as np
import copy

def uniform_crossover(parents):
    """ 
    Uniform crossover
    """
    mask = np.random.uniform(size=GAIndividual.gene_count) < 0.5
    offspring = copy.deepcopy(parents)
    offspring[0][mask] = parents[1][mask]
    offspring[1][mask] = parents[0][mask]
    return offspring

def normal_mutation(chromosome, sigma = 0.1):
    """
    Mutates random genes chosen from uniform dist.
    - Adds number from std.normal dist to gene.
    - sigma: standard deviation
    """
    mask = np.random.uniform(size=GAIndividual.gene_count) < 0.5 # which genes to manipulate
    chromosome[mask] += sigma * np.random.randn( GAIndividual.gene_count )[mask]
    return chromosome