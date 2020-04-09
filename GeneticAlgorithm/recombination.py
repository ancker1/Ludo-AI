from individual import GAIndividual
import numpy as np
import copy

def uniform_crossover(parents):
    """ 
    Uniform crossover
    """
    mask = np.random.uniform(size=GAIndividual.gene_count) < 0.5
    offspring = copy.deepcopy(parents)
    offspring[0] = parents[1][mask]
    offspring[1] = parents[0][mask]
    return offspring

def normal_mutation(chromosome):
    """
    Mutates random genes chosen from uniform dist.
    - Adds number from std.normal dist to gene.
    """
    mask = np.random.uniform(size=GAIndividual.gene_count) < 0.5 # which genes to manipulate
    chromosome[mask] += np.random.randn( GAIndividual.gene_count )
    return chromosome