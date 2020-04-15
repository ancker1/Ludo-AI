import numpy as np

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
        action_values = [-10000] * len(next_states)
        for i in range(len(next_states)):
            if next_states[i] != False:
                action_values[i] = np.dot(self.evaluate_action(state, next_states[i], i), self.chromosome)
        #action_values = [np.dot(self.evaluate_action(state, next_states[i], i), self.chromosome) for i in range(4)]
        return np.argmax(action_values)

    def play(self, state, dice_roll, next_states):
        """ Return action that evaluated to max value """
        return self.evaluate_actions(state, next_states)


class GANNIndividual:
    """
    Genetic Algorithm: Individual (X genes, value encoding)
    """
    name = 'NN Individual'

    input_size  = 4 * 4 # 4 tokens per player
    output_size = 4     # only 4 actions
    hidden_neurons = 20 # 20 hidden neurons
    gene_count = hidden_neurons * input_size + hidden_neurons + hidden_neurons * output_size + output_size

    def __init__(self):
        self.hidden_neurons = 20
        self.input_size     = 4 * 4
        self.output_size    = 4

        self.W1 = np.zeros((self.hidden_neurons, self.input_size))
        self.b1 = np.zeros((self.hidden_neurons, 1))
        self.W2 = np.zeros((self.output_size, self.hidden_neurons))
        self.b2 = np.zeros((self.output_size, 1))
            
    

    def load_chromosome(self, chromosome):
        """
        Chromosome contains weights for NN
        """
        self.chromosome = chromosome
        w1_siz = self.hidden_neurons*self.input_size
        w2_siz = self.output_size*self.hidden_neurons
        self.W1 = self.chromosome[0 : w1_siz].reshape((self.hidden_neurons, self.input_size))
        self.W2 = self.chromosome[w1_siz : w1_siz + w2_siz].reshape((self.output_size, self.hidden_neurons))
        self.b1 = self.chromosome[w1_siz + w2_siz : w1_siz + w2_siz + self.hidden_neurons].reshape((self.hidden_neurons, 1))
        self.b2 = self.chromosome[w1_siz + w2_siz + self.hidden_neurons : w1_siz + w2_siz + self.hidden_neurons + self.output_size].reshape((self.output_size, 1))


    # tanh
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    # softmax
    @staticmethod
    def softmax(x):
        return x

    def forward(self, chromosome, state):
        net_input = np.ravel(state)
        activation = np.dot(self.W1, net_input) + self.b1
        activation = self.tanh(activation)

        output = np.dot(self.W2, activation) + self.b2
        #output = softmax(output)
        return output
        

    def evaluate_actions(self, state, next_states):
        action_values = forward(self.chromosome, state)
        action_values[next_states == False] = -1e6
        return np.argmax(action_values)

    def play(self, state, dice_roll, next_states):
        """ Return action that evaluated to max value """
        return self.evaluate_actions(state, next_states)
    
