import sys
sys.path.append('../../pyludoperf/')
from pyludo.utils import token_vulnerability, star_jump, will_send_self_home, will_send_opponent_home, is_globe_pos
import random
import numpy as np

class LudoPlayerRandom:
    """ takes a random valid action """
    name = 'random'

    #@staticmethod
    #def play(state, dice_roll, next_states):
    #    return random.choice(np.argwhere(next_states != False))
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

class GreedyQLearner:
    """ Uses Learned Q-Dict """
    name = 'Greedy Q-learner'

    def __init__(self, Q_DICT):
        self._Q = Q_DICT
        
        self._default = lambda : None

        self._default.initialQ = 0
        self._default.stateSize = 4
        self._default.amountActions = 4

        self._state = lambda : None
        self._state.default     = -1# 0
        self._state.normal      = 0 # 1
        self._state.home        = 1 # 10
        self._state.safe        = 2 # 100
        self._state.endgame     = 3 # 1000
        self._state.vulnerable  = 4 # 10000
        self._state.goal        = 5 # 100000
        self._state.stacked     = 6 # 1000000
        self._state.globe       = 7 # 10000000
        self._state.homeglobe   = 8 # 100000000
        # self._state.canKill

        self._action = lambda : None
        self._action.default    = 0
        self._action.normal     = 1
        self._action.suicide    = 2
        self._action.kill       = 3
        self._action.safe       = 4
        self._action.endgame    = 5
        self._action.jump       = 6
        self._action.goal       = 7

        self._policy = self._createGreedyPolicy()
        
    def _getUniqueStateIdentifier(self, state, dice_roll = -1):
        """
        Translates state into a unique string identifier.

        If state have not been visited before, then create entry in self._Q
        """
        # stateIdentifier = []
        # [stateIdentifier.append(str(np.sort(st))) for st in state]  # np.sort() is used to create unique identifier
        # stateIdentifier = ''.join(stateIdentifier)                  # sorted since it does not matter which brick is located where
        # if not stateIdentifier in self._Q:
        #     self._Q[stateIdentifier] = [self._default.initialQ] * self._default.amountActions
        tstate = self._translateState(state)
        tstate.append(dice_roll)
        stateIdentifier = str(tstate)
        if not stateIdentifier in self._Q:
            self._Q[stateIdentifier] = [self._default.initialQ] * self._default.amountActions
        return stateIdentifier     
            
    def _translateState(self, state):
        """
        Translate state into a simplification
        """
        if type(state) == bool:
            return False

        tstate = [self._state.default] * self._default.stateSize
        for i in range(len(state[0])):
            pos = state[0][i]
            
            # create vulnerable state <<<<<<<==================================================================
            if pos == -1:
                tstate[i] = self._state.home    # home
                continue    # if at home no other state can be present
            
            if -1 < pos and pos < 52:
                tstate[i] = self._state.normal  # nothing special

            if token_vulnerability(state, i):
                tstate[i] = self._state.vulnerable    # vulnearable

            if pos < 51:
                if np.sum( state[0][i] == pos ) > 1:
                    tstate[i] = self._state.stacked
                if is_globe_pos(pos):
                    tstate[i] = self._state.homeglobe
                if is_globe_pos(pos) and pos % 13 != 1:
                    tstate[i] = self._state.globe

            if pos > 51:
                tstate[i] = self._state.endgame # permanently safe

            if pos == 99:
                tstate[i] = self._state.goal    # goal
        return tstate
            

    def _createGreedyPolicy(self):
        """
        This should be used when exploiting what have been learned during training.
        """
        def policy(state, next_states):
            probabilities = np.array([0] * self._default.amountActions)
            bestAction = np.argmax( self._Q[state] )
            i = self._default.amountActions - 1
            while next_states[bestAction] == False and (i >= 0):
                bestAction = np.argsort( self._Q[state] )[i]
                i -= 1
            probabilities[bestAction] = 1
            return probabilities
        return policy

    def _createEpsilonGreedyPolicy(self, epsilon):
        """ Creates an epsilon-greedy policy
        
            Returns policy which with a probability of (1 - epsilon) chooses the best action and with probability epsilon chooses random.
         """
        def policy(state, next_states):
            """
            Inputs are state and next_states

            next_states are needed to check for valid moves and set probabilities of being chosen accordingly
            """
            probabilities = np.ones( self._default.amountActions ) * epsilon / ( self._default.amountActions - np.count_nonzero(next_states == False))
            probabilities[np.where(next_states == False)] = 0
            bestAction = np.argmax( self._Q[state] )
            i = self._default.amountActions - 1
            while next_states[bestAction] == False and (i >= 0):
                bestAction = np.argsort( self._Q[state] )[i]
                i -= 1
            
            probabilities[bestAction] += ( 1 - epsilon )        
            
            return probabilities

        return policy

    def _getHighestActionValue(self, nextState):
        state = self._getUniqueStateIdentifier(nextState)
        return np.max( self._Q[state] )
    
    def play(self, state, dice_roll, next_states):
        #print("Dice: "+str(dice_roll)+", state: "+str(state[0]))
        #print(self._translateState(state))
        
        currentState    = self._getUniqueStateIdentifier(state, dice_roll=dice_roll) # translate state into a unique state identifier (sorted states)
        probabilities   = self._policy(currentState, next_states)   # calculate probabilities derived from policy (probabilities is wrt. sorted states)
        action          = np.random.choice(range(probabilities.size), p = probabilities)         # choose action with respect to the policy (chosen wrt. sorted states)
        
        return action # action