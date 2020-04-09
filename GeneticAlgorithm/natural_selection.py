from population import GAPopulation
from individual import GAIndividual
from recombination import uniform_crossover, normal_mutation
from selection import rank_selection
from replacement import elitism
import numpy as np
import argparse



class Evolution:
    def __init__(self):
        self.selection          = rank_selection
        self.genetic_operator   = uniform_crossover
        self.mutator            = normal_mutation
        self.replacement        = elitism
        self.pc = 0.05 # probability of mutation
        self.generation_count   = 0

    def recombination(self, pairs):
        offspring = self.crossover(pairs)
        return self.mutation(offspring)

    def mutation(self, offspring):
        mask = np.random.uniform(size = len(offspring)) < self.pc
        return [self.mutator(offspring[i]) if mask[i] else offspring[i] for i in range(len(offspring))]

    def crossover(self, pairs):
        return np.array([self.genetic_operator(parents) for parents in pairs]).reshape(len(pairs)*2) # flatten from [[child_1, child_2],...] to [child_1, child_2, ....]


## READ ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--offsprings', type=int, default=10) # read by: args.offsprings
parser.add_argument('--generations', type=int, default=10)
args = parser.parse_args()

if __name__ == "__main__":
    evolution = Evolution()

    population = GAPopulation()
    population.evaulate_fitness_against_pop()

    while evolution.generation_count < args.generations:
        pairs = evolution.selection(population, args.offsprings)
        offspring = evolution.recombination(pairs)
        population = evolution.replacement(population, offspring)
        population.evaulate_fitness_against_pop()
        evolution.generation_count += 1

    
