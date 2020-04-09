from evaluate_chromosome import evaluate_agent, evaluate_agents
from tqdm import tqdm
import numpy as np
import random

class GAPopulation:
    """ Genetic Algorithm: Population """
    name = 'Population'
    
    def __init__(self):
        self.population_size = 20
        self.evaluations_per_chromosome = 25
        self.individual = GAIndividual()
        self.init_pop()
        self.fitness = [0] * self.population_size

    def init_pop(self):
        """Initialize population according to standard normal distribution.
            Genes are afterwards normalized into a range of [0, 1]."""
        genes = np.random.randn( self.population_size * self.individual.gene_count )
        self.population = genes.reshape((self.population_size, -1))
        for i, chromosome in enumerate(self.population):
            self.population[i] = (chromosome - np.min(chromosome))/np.ptp(chromosome)
        print(self.population)

    def evaluate_fitness_against_random(self):
        """ Evaluate fitness of population - with evaluation against RANDOM """
        for i, chromosome in enumerate(self.population):
            print('Fitness: '+str(i))
            self.individual.load_chromosome(chromosome)
            self.fitness[i] = evaluate_agent(self.individual, 100)
        print(self.fitness)

    def evaulate_fitness_against_pop(self):
        """ Evaluate fitness of population - with evaluation against AGENTS """
        self.fitness    = [0] * self.population_size  # Reset fitness
        evaluations     = [0] * self.population_size
        agents = [GAIndividual(), GAIndividual(), GAIndividual(), GAIndividual()]
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

class GAIndividual:
    """ Genetic Algorithm: Individual (4 genes, value encoding) """
    name = 'Individual'

    gene_count = 4

    def load_chromosome(self, chromosome):
        """ chromosome is value encoded: weighting of events """
        self.chromosome = chromosome

    @staticmethod
    def opponent_home_tokens(state):
        """ Return amount of opponents tokens that is in their home given a state """
        tokens = 0
        for opponent in state[1:]:
            for token in opponent:
                if token == -1:
                    tokens += 1
        return tokens

    def evaluate_action(self, state, next_state, i):
        """ Return array of events """
        if next_state == False:
            return np.array([-1000] * GAIndividual.gene_count)

        cur_pos     =   state[0][i]
        next_pos    =   next_state[0][i]

        moved_onto_board    = ( next_pos > -1 ) and ( cur_pos == -1 )
        send_opponent_home  = self.opponent_home_tokens(next_state) > self.opponent_home_tokens(state)
        end_zone            = ( next_pos > 51  ) and ( cur_pos <= 51 )
        goal                = ( next_pos == 99 ) and ( cur_pos != 99 )
        return np.array([moved_onto_board, send_opponent_home, end_zone, goal])

    def evaluate_actions(self, state, next_states):
        """ Find value of all actions -> choose arg max """
        action_values = [np.dot(self.evaluate_action(state, next_states[i], i), self.chromosome) for i in range(4)]
        return np.argmax(action_values)

    def play(self, state, dice_roll, next_states):
        """ Return action that evaluated to max value """
        return self.evaluate_actions(state, next_states)