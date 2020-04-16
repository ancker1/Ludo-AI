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

