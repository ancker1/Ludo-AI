import numpy as np
import random

class LudoPlayerRandom:
    """ takes a random valid action """
    name = 'random'

    @staticmethod
    def play(state, dice_roll, next_states):
        return random.choice(np.argwhere(next_states != False))