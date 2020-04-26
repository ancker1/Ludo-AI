import sys
sys.path.append('../../pyludoperf/')
from pyludo.utils import token_vulnerability, star_jump, will_send_self_home, will_send_opponent_home, is_globe_pos
import random
import numpy as np

class QSimple:
    """ Uses Q-Learning """
    name = 'Q Simple'

    def __init__(self, standardMoveType = 0):
        self.alpha      = 0.1
        self.gamma      = 0.2
        self.epsilon    = 0.1
        state = 0
        # can move onto board, can reach endgame, can send opp. home, can reach goal
        self.stateMap = [1, 2, 4, 8]
        # move onto board, reach endgame, send opp. home, reach goal, standard move
        self.actionMap = [0, 1, 2, 3, 4]
        # 4 bits describing state 
        self.stateBits = [0, 0, 0, 0]
        self.state     = 0
        # used for update
        self.cstate     =  -1 # Needed for initial self.cstate
        self.reward     =  0
        self.action     =  0
        self.QAction    =  0
        self.nstate     =  lambda : None
        self.Q          =  np.zeros((16, 5))
        # utils
        self.accumulatedReward = 0
        self.moves = 0
        # Exploit knowledge
        self.actGreedy = False
        # Set Standard move
        if standardMoveType == 0:
            self.standardMove = self.moveClosestToGoal
        elif standardMoveType == 1:
            self.standardMove = self.moveRandom
        else:
            self.standardMove = self.moveResultingLargestStep

    def getState(self, state, next_states):
        self.stateBits = [0] * 4
        self.state     =  0
        for nstate in next_states:
            if nstate is not False:
                # check if any nstate leads to: move onto board
                if np.sum(np.array(state.state[0]) == -1) > np.sum(np.array(nstate.state[0]) == -1):
                    self.stateBits[0] = 1
                # check if any nstate leads to: reach endgame zone
                if np.sum(np.array(state.state[0]) < 51) > np.sum(np.array(nstate.state[0]) < 51):
                    self.stateBits[1] = 1
                # check if any nstate leads to: send opponent home
                if np.sum(np.array(state.state[1:]) == -1) < np.sum(np.array(nstate.state[1:]) == -1):
                    self.stateBits[2] = 1
                # check if any nstate leads to: reach goal
                if np.sum(np.array(state.state[0]) == 99) < np.sum(np.array(nstate.state[0]) == 99):
                    self.stateBits[3] = 1
        # collapse state bits into integer
        for i, bit in enumerate(self.stateBits):
            self.state += self.stateMap[i] * bit
        return self.state

    def moveClosestToGoal(self, state, next_states):
        sortedChoices = np.argsort(state.state[0])[::-1]
        for i in range(4):
            if next_states[sortedChoices[i]] is not False:
                return sortedChoices[i]

    def moveRandom(self, state, next_states):
        for num in random.sample(list(range(4)), 4):
            if next_states[num] is not False:
                return num

    def moveResultingLargestStep(self, state, next_states):
        # Move token that leads to largest step
        max_indice = 0
        max_step   = -2000
        for i, nstate in enumerate(next_states): # Move token that takes the largest step
            if nstate is not False:
                current_step = nstate[0][i] - state.state[0][i]
                if current_step > max_step:
                    current_step = max_step
                    max_indice = i
        return max_indice
   
    def getGreedyChoice(self):
        sortedArgs = np.argsort(self.Q[self.state])[::-1]
        i = 0
        invalidMove = True
        while invalidMove: # Raise prob. of best valid action
            if sortedArgs[i] == 4:
                return 4
                invalidMove = False
            elif self.stateBits[sortedArgs[i]] == 1:
                return sortedArgs[i]
                invalidMove = False
            i += 1
        return -1

    def evaluateActions(self):
        # 4 special actions and 1 standard
        probabilities = np.ones( 5 ) * self.epsilon / (sum(self.stateBits) + 1) # Divide by the amount of valid actions
        # Find valid actions
        probabilities[[i for i, val in enumerate(self.stateBits) if val == 0]] = 0 # Set invalid actions to 0
        if sum(self.stateBits) == 0:
            probabilities[4] += (1 - self.epsilon)
        else:
            sortedArgs = np.argsort(self.Q[self.state])[::-1]
            i = 0
            invalidMove = True
            while invalidMove: # Raise prob. of best valid action
                if sortedArgs[i] == 4:
                    probabilities[4] += (1 - self.epsilon)
                    invalidMove = False
                elif self.stateBits[sortedArgs[i]] == 1:
                    probabilities[sortedArgs[i]] += (1 - self.epsilon)
                    invalidMove = False
                i += 1
        return np.random.choice(range(probabilities.size), p = probabilities) # Get action from probabilities

    def getTokenToMove(self, QAction, state, next_states):
        if QAction == 0:
            # Return token leading to: move onto board
            for i, nstate in enumerate(next_states):
                if nstate is not False:
                    if np.sum(np.array(state.state[0]) == -1) > np.sum(np.array(nstate.state[0]) == -1):
                        # found token move leading to move onto board
                        return i
        elif QAction == 1:
            # Return token leading to: Reach endgame zone
            for i, nstate in enumerate(next_states):
                if nstate is not False:
                    if np.sum(np.array(state.state[0]) < 51) > np.sum(np.array(nstate.state[0]) < 51):
                        # found token move leading to reach endgame
                        return i
        elif QAction == 2:
            # Return token leading to: send opponent home
            for i, nstate in enumerate(next_states):
                if nstate is not False:
                    if np.sum(np.array(state.state[1:]) == -1) < np.sum(np.array(nstate.state[1:]) == -1):
                        # found token move leading to send opponent home
                        return i
        elif QAction == 3:
            # Return token leading to: reach goal
            for i, nstate in enumerate(next_states):
                if nstate is not False:
                    if np.sum(np.array(state.state[0]) == 99) < np.sum(np.array(nstate.state[0]) == 99):
                        # found token move leading to reach goal
                        return i
        elif QAction == 4:
            # Return standard move
            return self.standardMove(state, next_states)

        return -1 # ERROR

    def getReward(self, state, nstate, token):
        ''' 
        Evaluate which event(s) actually takes place
        '''
        # Multiple rewards may be returned
        reward = 0
        if np.sum(np.array(state.state[0]) == -1) > np.sum(np.array(nstate.state[0]) == -1):
            # Move onto board reward
            reward += 0.5
        if np.sum(np.array(state.state[0]) < 51) > np.sum(np.array(nstate.state[0]) < 51):
            # Reach endgame reward
            reward += 1
        if np.sum(np.array(state.state[1:]) == -1) < np.sum(np.array(nstate.state[1:]) == -1):
            # Send opponent home reward
            reward += 0.5
        if np.sum(np.array(state.state[0]) == 99) < np.sum(np.array(nstate.state[0]) == 99):
            # move into goal reward
            reward += 1
        if ( np.array(nstate.state[0])[token] % 13 == 6 or np.array(nstate.state[0])[token] % 13 == 12 ) and not ( np.array(nstate.state[0])[token] == 1 or np.array(nstate.state[0])[token] > 51 ):
            # star jump
            reward += 0.5
        return reward


    def getMaxQ(self, nstate):
        return np.max(self.Q[nstate, :])

    def play(self, state, dice_roll, next_states):
        if self.actGreedy:
            cstate = self.getState(state, next_states)
            return self.getTokenToMove(self.getGreedyChoice(), state, next_states)

        self.moves += 1
        nstate = self.getState(state, next_states) # use current state as nstate in update
        # RL needs to observe reward before it can update
        if self.cstate != -1:                 # use previous state as cstate in update
            # When reward is obtained -> everything is known -> update can be performed
            #self.Q[self.cstate, self.QAction] = 1
            #print(self.reward)
            self.Q[self.cstate, self.QAction] += self.alpha * ( self.reward + self.gamma * self.getMaxQ(nstate)  - self.Q[self.cstate, self.QAction] )
        
        self.cstate     =   nstate  # Save state to use as cstate in next update step
        self.QAction    =   self.evaluateActions() # Compute which QAction is chosen
        self.action     =   self.getTokenToMove(self.QAction, state, next_states)  # Compute which token to move based on QAction
        # Compute reward based on the chosen action
        self.reward = self.getReward(state, next_states[self.action], self.action) # Compute reward based on which state nstate actually became
        self.accumulatedReward += self.reward
        return self.action # Return the chosen token to move

class GreedyQLearner:
    """ Uses Learned Q-Dict """
    name = 'Q Advanced'

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