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