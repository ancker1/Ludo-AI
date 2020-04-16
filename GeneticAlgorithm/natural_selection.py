from population import GAPopulation
from individual import GAIndividual, GANNIndividual
from recombination import uniform_crossover, normal_mutation
from selection import rank_selection
from replacement import elitism
import numpy as np
import argparse



class Evolution:
    def __init__(self, mutation_rate, type_index):
        if type_index == 0:
            self.individual_type = GAIndividual
        else:
            self.individual_type = GANNIndividual

        self.selection          = rank_selection
        self.genetic_operator   = uniform_crossover
        self.mutator            = normal_mutation
        self.replacement        = elitism
        self.pc = mutation_rate # probability of mutation
        self.generation_count   = 0

    def recombination(self, pairs, chromosomes):
        offspring = self.crossover(pairs, chromosomes)
        return self.mutation(offspring)

    def mutation(self, offspring):
        mask = np.random.uniform(size = len(offspring)) < self.pc
        return [self.mutator(offspring[i], individual_type=self.individual_type) if mask[i] else offspring[i] for i in range(len(offspring))]

    def crossover(self, pairs, chromosomes):
        offspring = np.array([self.genetic_operator([chromosomes[parents[0]], chromosomes[parents[1]]], individual_type=self.individual_type) for parents in pairs])
        return offspring.reshape(len(pairs)*2, -1) # flatten from [[child_1, child_2],...] to [child_1, child_2, ....]

    def save_population(self, chromosomes):
        filepath = "data/gen{}.npy".format(str(self.generation_count))
        np.save(filepath, population.get_chromosomes())


## READ ARGS
parser = argparse.ArgumentParser()
parser.add_argument('--offsprings',     type=float, default = 0.8   ) # read by: args.offsprings
parser.add_argument('--generations',    type=int,   default = 10    )
parser.add_argument('--elitism',        type=float, default = 0.2   )
parser.add_argument('--mutation_rate',  type=float, default = 1     )
parser.add_argument('--individual',     type=int,   default = 0     )
args = parser.parse_args()

if __name__ == "__main__":
    evolution = Evolution(args.mutation_rate, args.individual)

    population = GAPopulation( evolution.individual_type )
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
    
