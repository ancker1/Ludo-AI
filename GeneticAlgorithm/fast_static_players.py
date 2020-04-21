import numpy as np
import random

class LudoPlayerRandom:
    """ takes a random valid action """
    name = 'random'

    @staticmethod
    def play(state, dice_roll, next_states):
        for num in random.sample(list(range(4)), 4):
            if next_states[num] is not False:
                return num

class SemiSmartPlayer:
    """ Semi smart player that follows a static strategy """
    name = 'semismart'

    def play(self, state, dice_roll, next_states):
        for i, nstate in enumerate(next_states):
            if nstate is not False: # Move onto board if possible - > kill if possible
                if np.sum(np.array(state.state[1:]) == -1) < np.sum(np.array(nstate.state[1:]) == -1):
                    return i
                if np.sum(np.array(state.state[0]) == -1) > np.sum(np.array(nstate.state[0]) == -1):
                    return i
        max_indice = 0
        max_step   = -2000
        for i, nstate in enumerate(next_states): # Move token that takes the largest step
            if nstate is not False:
                current_step = nstate[0][i] - state.state[0][i]
                if current_step > max_step:
                    current_step = max_step
                    max_indice = i
        return max_indice

