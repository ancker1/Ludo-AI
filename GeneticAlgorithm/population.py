from evaluate_chromosome import evaluate_agent, evaluate_agents
from individual import GAIndividual, GANNIndividual
from tqdm import tqdm
import numpy as np
import random

class GAPopulation:
    """ Genetic Algorithm: Population """
    name = 'Population'
    
    def __init__(self, individual_type, adjust=0, population_size = 100, adjust_size=20):
        self.adjust             = adjust
        self.adjust_size        = adjust_size
        self.population_size    = population_size
        self.evaluations_per_chromosome = 25

        self.individual_type    = individual_type
        self.individual         = self.individual_type()
        self.fitness            = [0] * self.population_size

        self.init_pop()
        self.normalize()

    def load_chromosomes(self, chromosomes):
        self.population = chromosomes

    def normalize(self, type_range = 0):
        """ Normalize genes into range
        (type_range: 0) [-1, 1]
        (type_range: 1) [0,1 ]"""
        
        for i, chromosome in enumerate(self.population):
            self.population[i] = chromosome / np.sum(np.abs(chromosome)) * self.individual_type.gene_count

        #for i, chromosome in enumerate(self.population):
        #    if type_range == 0:
        #        self.population[i] = 2 * (chromosome - np.min(chromosome))/np.ptp(chromosome) - 1
        #    elif type_range == 1:
        #        self.population[i] = (chromosome - np.min(chromosome))/np.ptp(chromosome)

    def init_pop(self):
        """Initialize population according to standard normal distribution.
            Genes are afterwards normalized into a range of [0, 1]."""
        genes = np.random.randn( self.population_size * self.individual.gene_count )
        self.population = genes.reshape((self.population_size, -1))
        #print(self.population)

    def shrink_population(self):
        # Requires population with loaded chromosomes with fitness against pop.
        self.evaulate_fitness_against_pop()
        print("Before shrink:",self.population.shape)
        self.population = self.population[np.argsort(self.fitness)[::-1][:self.adjust_size]]
        print("After shrink:",self.population.shape)
        self.population_size = self.adjust_size
        self.fitness         = [0] * self.population_size

    def evaluate_fitness(self):
        if self.individual_type == GAIndividual:
            self.normalize() # Normalize before evaluating

        if self.adjust == 0:
            self.evaulate_fitness_against_pop()
        else:
            self.evaluate_fitness_against_random()

    def evaluate_fitness_against_specific(self, use_random):
        #self.normalize() # Normalize before evaluating
        for i in tqdm(range(self.population_size)):
            self.individual.load_chromosome(self.population[i])
            self.fitness[i] = evaluate_agent(self.individual, self.evaluations_per_chromosome * 4, use_random=use_random) / (self.evaluations_per_chromosome * 4)
        print(self.fitness)

    def evaluate_fitness_against_random(self):
        """ Evaluate fitness of population - with evaluation against RANDOM """
        #self.normalize() # Normalize before evaluating
        for i in tqdm(range(self.population_size)):
            self.individual.load_chromosome(self.population[i])
            self.fitness[i] = evaluate_agent(self.individual, self.evaluations_per_chromosome * 4) / (self.evaluations_per_chromosome * 4)
        print(self.fitness)

    def evaulate_fitness_against_pop(self):
        """ Evaluate fitness of population - with evaluation against AGENTS """
        #self.normalize() # Normalize before evaluating

        self.fitness    = [0] * self.population_size  # Reset fitness
        evaluations     = [0] * self.population_size
        agents = [self.individual_type(), self.individual_type(), self.individual_type(), self.individual_type()]
        for _ in tqdm(range(self.population_size * self.evaluations_per_chromosome)):
            sample = [random.randint(0,self.population_size - 1) for _ in range(4)] # sample random chromosomes
            for i in range(len(sample)):
                evaluations[sample[i]] += 1
                agents[i].load_chromosome(self.population[sample[i]]) # Load chromosome of random sample [i] into agent[i]
            win_index = evaluate_agents(agents)
            self.fitness[sample[win_index]] += 1        # increment fitness of winner -> amount of wins is obtained
        for i, fitness in enumerate(self.fitness):
            self.fitness[i] = fitness / evaluations[i]  # Divide fitness with amount of times chromosome have been evaluated -> win rate is obtained.

    def get_best_chromosome(self):
        return self.population[np.argmax(self.fitness)]

    def get_chromosomes(self):
        return self.population