import sys
sys.path.append('../../pyludo/')
from pyludo.utils import token_vulnerability, star_jump, will_send_self_home, will_send_opponent_home, is_globe_pos
from pyludo import LudoGame
import matplotlib.pyplot as plt
from os import path
from playes import LudoPlayerRandom
from tqdm import tqdm
import numpy as np
import random
import time
import csv
    
class LudoQLearner:
    """ Uses Q-Learning """
    name = 'Q-learner'

    def __init__(self, Q_DICT, learning_rate = 0.1, discount_factor = 0.2, training=True, _type=0):
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._Q = Q_DICT
        self._training = training

        self._default = lambda : None

        self._default.epsilon = 0.1
        self._default.initialQ = 0
        self._default.stateSize = 4
        self._default.amountActions = 4

        self._default.rewards = [0, 0, -50, 50, 50, 50, 0, 100]
        self.accumulated_reward = 0.0

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

        if _type == 0:
            self._default.rewards = [0, 0, -50, 50, 50, 50, 0, 100]
        if _type == 1:
            self._default.rewards = [0, 0, -50, 10, 10, 100, 0, 50]
        if _type == 2:
            self._default.rewards = [0, 0, -100, 100, 100, 100, 0, 10]
        if _type == 3:
            self._default.rewards = [0, 0, -100, 0, 0, 0, 0, 100]

        if self._training:
            self._policy = self._createEpsilonGreedyPolicy(self._default.epsilon)
        else:
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

    def _getEvents(self, state, next_state):
        events = [] * self._default.amountActions
        for i in range(self._default.amountActions): # len(next_state)
            if type(next_state) == bool:
                events = [None] # resulting state is invalid - no actions possible

            tstate = self._translateState(next_state)
            nextPos = next_state[0][i]

            if tstate[i] == self._state.normal: # action is a "normal" move
                events.append(self._action.normal)

            if nextPos < 53:
                if np.sum( next_state[0][i] == nextPos ) > 1 or (is_globe_pos(nextPos) and nextPos % 13 != 1 ) or nextPos % 13 == 1 and np.sum(next_state[nextPos // 13] == -1) == 0:    
                    events.append(self._action.safe) # action is two on same spot OR on globe (not outside home) OR on globe outside empty home

            if star_jump(nextPos): # jump action
                events.append(self._action.jump)

            if state[0][i] < 52 and nextPos > 52:   # go into endgame
                events.append(self._action.endgame)

            if will_send_self_home(state, next_state):     # suicide
                events.append(self._action.suicide)

            if will_send_opponent_home(state, next_state): # send opponent home
                events.append(self._action.kill)

            if state[0][i] != 99 and nextPos == 99:
                events.append(self._action.goal)

        return events




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

    def _getReward(self, state, action, dice_roll): 
        newState = state.move_token(action, dice_roll) # find next state
        reward = 0.0

        events = self._getEvents(state, newState)      # get events triggered by action
        for event in events:
            reward += self._default.rewards[event]     # accumulate reward

        return (reward, newState)

    def _getHighestActionValue(self, nextState):
        state = self._getUniqueStateIdentifier(nextState)
        return np.max( self._Q[state] )

    def play(self, state, dice_roll, next_states):
        #print("Dice: "+str(dice_roll)+", state: "+str(state[0]))
        #print(self._translateState(state))

        currentState    = self._getUniqueStateIdentifier(state, dice_roll=dice_roll) # translate state into a unique state identifier (sorted states)
        probabilities   = self._policy(currentState, next_states)   # calculate probabilities derived from policy (probabilities is wrt. sorted states)
        action          = np.random.choice(range(probabilities.size), p = probabilities)         # choose action with respect to the policy (chosen wrt. sorted states)

        if self._training:
            # Get (reward & next state) from current state and action
            (reward, nextState) = self._getReward(state, action, dice_roll)
            self.accumulated_reward += reward
            # Update the _Q value from state and action
            self._Q[currentState][action] += self._learning_rate * ( reward + self._discount_factor * self._getHighestActionValue(nextState)  - self._Q[currentState][action] )

        return action # action

Q_DICT = dict()

########################################################
#                SET TRAINING STATE                    #
########################################################
TRAINING = False                                       #
########################################################
if path.isfile("experience.csv"):
    experience = csv.reader(open("experience.csv"))
    for row in experience:
        if row:
            k,v = row
            v = np.fromstring(v[1:-1], sep=',').tolist()
            Q_DICT[k] = v

if TRAINING:
    trainingRewards = []
    if path.isfile("rewards.csv"):
        rewards = csv.reader(open("rewards.csv"))
        for row in rewards:
            if row:
                trainingRewards.append(float(row[0]))

agent = LudoQLearner(Q_DICT, training=TRAINING)
evaluator = LudoPlayerRandom()
players = [ agent, agent, evaluator, evaluator  ]

for i, player in enumerate(players):
    player.id = i

wins = [0, 0, 0, 0]

N = 1000

start_time = time.time()
for i in tqdm(range(N)):
    #print("Dict size: "+str(len(Q_DICT)))
    random.shuffle(players)
    
    if TRAINING:
        for p in players:
            if type(p) == LudoQLearner:
                trainingRewards.append(p.accumulated_reward)
                p.accumulated_reward = 0.0

    ludoGame = LudoGame(players)
    winner = ludoGame.play_full_game()
    wins[players[winner].id] += 1
    #if i % 100 == 0:
        #print('Game ', i, ' done')
duration = time.time() - start_time

if TRAINING:
    w = csv.writer(open("experience.csv", "w"))
    for key, val in Q_DICT.items():
        if val != [0] * 4:
            w.writerow([key, val])

    rewardFile = csv.writer(open("rewards.csv", "w"))
    for reward in trainingRewards:
        rewardFile.writerow([reward])

print('win distribution:', wins)
print('games per second:', N / duration)