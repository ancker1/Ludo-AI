from individual import GAIndividual, GANNIndividual
import numpy as np
import copy

def uniform_crossover(parents, individual_type = GAIndividual, crossover_ratio=0.85):
    """ 
    Uniform crossover
    """
    # if np.random.uniform() > crossover_ration:
    #   return parents
    mask = np.random.uniform(size=individual_type.gene_count) < 0.5
    offspring = copy.deepcopy(parents)
    offspring[0][mask] = parents[1][mask]
    offspring[1][mask] = parents[0][mask]
    return offspring

def normal_mutation(chromosome, individual_type = GAIndividual, sigma = 0.15, mutate_all = True):
    """
    Mutates random genes chosen from uniform dist.
    - Adds number from std.normal dist to gene.
    - sigma: standard deviation
    """
    if mutate_all:
        mask = [True] * individual_type.gene_count
    else:
        mask = np.random.uniform(size=individual_type.gene_count) < 0.5 # which genes to manipulate
    chromosome[mask] += sigma * np.random.randn( individual_type.gene_count )[mask]
    return chromosome