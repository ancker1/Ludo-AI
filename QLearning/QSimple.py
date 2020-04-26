import sys
sys.path.append('../../pyludoperf/')
from pyludo import LudoGame
import numpy as np
import random

class QSimple:
    """ Uses Q-Learning """
    name = 'Q-learner'

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