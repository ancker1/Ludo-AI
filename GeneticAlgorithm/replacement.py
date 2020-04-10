from population import GAPopulation
import numpy as np

def elitism(population : GAPopulation, offspring, n_best = 10):
    """
    Keep n_best individuals and replace the rest with offspring
    """
    sorted_fitness = np.argsort(population.fitness)[::-1]
    new_generation = []
    chromosomes = population.get_chromosomes()
    for i in range(n_best): # get n_best chromosomes
        new_generation.append(chromosomes[sorted_fitness[i]])
    for i in range(population.population_size - n_best): # get population_size - n_best offspring
        new_generation.append(offspring[i])
    population.load_chromosomes(np.array(new_generation))
