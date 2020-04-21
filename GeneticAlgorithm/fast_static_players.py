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

    @staticmethod
    def get_semismart_action(state, next_states):
        return 0

    def play(self, state, dice_roll, next_states):
        return get_semismart_action(state, next_states)

