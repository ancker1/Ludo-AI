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

    def recombination(self, pairs, chromosomes):
        offspring = self.crossover(pairs, chromosomes)
        return self.mutation(offspring)

    def mutation(self, offspring):
        mask = np.random.uniform(size = len(offspring)) < self.pc
        return [self.mutator(offspring[i]) if mask[i] else offspring[i] for i in range(len(offspring))]

    def crossover(self, pairs, chromosomes):
        offspring = np.array([self.genetic_operator([chromosomes[parents[0]], chromosomes[parents[1]]]) for parents in pairs])
        return offspring.reshape(len(pairs)*2, -1) # flatten from [[child_1, child_2],...] to [child_1, child_2, ....]

    def save_population(self, chromosomes):
        filepath = "data/gen{}.npy".format(str(self.generation_count))
        np.save(filepath, population.get_chromosomes())


## READ ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--offsprings',  type=float, default=0.8) # read by: args.offsprings
parser.add_argument('--generations', type=int, default=10)
parser.add_argument('--elitism',     type=float, default=0.2)
args = parser.parse_args()

if __name__ == "__main__":
    evolution = Evolution()

    population = GAPopulation()
    population.evaulate_fitness_against_pop()
    evolution.save_population(population.get_chromosomes())

    while evolution.generation_count < args.generations:
        print("Current best fitness")
        print(np.max(population.fitness))
        # Select parents to mate
        pairs       = evolution.selection(population, int(args.offsprings*population.population_size))
        # Take the parent pairs -> crossover between pairs -> mutation
        offspring   = evolution.recombination(pairs, population.get_chromosomes())
        # Replacement to generate new population
        evolution.replacement(population, offspring, n_best=int(args.elitism*population.population_size))
        # Evaluate fitness for new population
        population.evaulate_fitness_against_pop()
        # Increment generation count
        evolution.generation_count += 1
        # Save the new population
        evolution.save_population(population.get_chromosomes())

# load databy: np.load("gen{}.npy".format(generation_index))
    
