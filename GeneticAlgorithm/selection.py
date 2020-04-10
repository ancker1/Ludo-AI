from population import GAPopulation
import numpy as np

def rank_selection(population : GAPopulation, pair_count, use_ranking = False):
    """ Rank selection 
        use_ranking: determines if ranking should be used.
        - If set to false probability will be proportional to fitness.
    """
    pairs = []
    total = sum(range(0,population.population_size + 1)) if use_ranking else sum(population.fitness)
    probabilities = [0] * population.population_size
    sorted_fitness =  np.argsort(population.fitness)[::-1] # Descending order (highest first)
    for i in range(population.population_size):
        probabilities[sorted_fitness[i]] = (population.population_size - i) / total if use_ranking else population.fitness[sorted_fitness[i]] / total
    for _ in range(pair_count):
        pairs.append(np.random.choice(range(population.population_size), 2, p = probabilities))
    return pairs # return pair_count pairs
        
# implement tournament
    