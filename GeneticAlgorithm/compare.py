import sys
sys.path.append('../../pyludoperf/')
from pyludo import LudoGame
from individual import GAIndividual, GANNIndividual
from fast_static_players import LudoPlayerRandom, SemiSmartPlayer
from population import GAPopulation
from tqdm import tqdm
from QLearner import GreedyQLearner, QSimple
import numpy as np
import argparse
import random
import time
import csv
from os import path
from RasmusKuus.V1.GeneticAlgorithms import GeneticAlgorithmsClass
from RasmusKuus.V2.GeneticAlgorithms import GeneticAlgorithmsClass2
from RasmusKuus.V3.GeneticAlgorithms import GeneticAlgorithmsClass3

def tanh(input):
    return np.tanh(input)

pops = 200
GA = GeneticAlgorithmsClass(57,3,57,1,tanh,pops)
GA.importWeigths(0,"RasmusKuus/V1/Gen-14-Wins-503-57-3-57-1-weigths.npy")

GA2 = GeneticAlgorithmsClass2(58,0,0,1,tanh,pops)
GA2.importWeigths(0,"RasmusKuus/V2/Gen-16-Wins-72-58-0-0-1-weigths.npy")

pops3 = 100
GA3 = GeneticAlgorithmsClass3(60,0,0,1,tanh,pops3)
GA3.importWeigths(0,"RasmusKuus/V3/Gen-1-Wins-910-60-0-0-1-weigths.npy")

class GAPlayer3:
    global GA3
    name = 'GA Ras3'
    index = 0
    def getGameState(self,state):
        output = np.zeros(59)
        for i in range(4):
            for j in range(4):
                if i == 0:
                    if (state[i][j] == 99):
                        output[58] += 1
                    elif state[i][j] == -1:
                        output[0] += 1
                    else:
                        output[state[i][j]+1] += 1

                        if output[state[i][j]+1] > 1 or state[i][j]%13 == 1 or state[i][j]%13 == 9:
                            output[state[i][j]+1] += 1


                if state[i][j] < 52 and state[i][j] != -1:
                    output[state[i][j]+1] -= 1
                  
        return output
    

    def evaluate_actions(self,state, next_states, dice_roll):
        action_values = np.zeros(4)
        actions = 0
        
        for i, next_state in enumerate(next_states):
            if next_state is False:
                action_values[i] = -1
            else:
                actions += 1

        if actions > 1:
            stateSum = np.sum(state[0])
            for i, next_state in enumerate(next_states):
                if next_state is False:
                    action_values[i] = -99999
                else:
                    dif = (np.sum(next_state.state[0] - stateSum))/56
                    AiInput = np.append(self.getGameState(next_state.state),dif)
                    action_values[i] = GA3.runModel(AiInput,GA3.getPopulation(self.index))

        return np.argmax(action_values)

    def play(self,state, dice_roll, next_states):
        return self.evaluate_actions(state, next_states, dice_roll)

class GAPlayer2:
    global GA2
    """ takes a random valid action """
    name = 'GA Ras2'
    index = 0
    def getGameState(self,state):
        output = np.zeros(58)
        for i in range(4):
            for j in range(4):
                if (i == 0):
                    if (state[i][j] == 99):
                        output[57] += 0.25
                    elif state[i][j] == -1:
                        output[0] += 0.25
                    elif output[state[i][j]] != 1:
                        output[state[i][j]] += 0.5
                if state[i][j] == -1:
                    continue
                elif state[i][j] < 53 and output[state[i][j]] != -1:
                    output[state[i][j]] -= 0.5
        return output
    

    def evaluate_actions(self,state, next_states, dice_roll):
        action_values = np.zeros(4)
        actions = 0
        
        for i, next_state in enumerate(next_states):
            if next_state is False:
                action_values[i] = -1
            else:
                actions += 1

        if actions > 1:
            for i, next_state in enumerate(next_states):
                if next_state is False:
                    action_values[i] = -99999
                else:
                    AiInput = self.getGameState(next_state.state)
                    action_values[i] = GA2.runModel(AiInput,GA2.getPopulation(self.index))

        return np.argmax(action_values)

    def play(self,state, dice_roll, next_states):
        return self.evaluate_actions(state, next_states, dice_roll)

class GAPlayer:
    global GA
    """ takes a random valid action """
    name = 'GA Ras'
    index = 0
    def getGameState(self,state):
        output = np.zeros(57)
        for i in range(4):
            for j in range(4):
                if state[i][j] == -1:
                    continue
                if (i == 0):
                    if (state[i][j] == 99):
                        output[56] += 0.25
                    else:
                        output[state[i][j]-1] += 0.5
                elif state[i][j] < 52:
                    output[state[i][j]-1] -= 0.5
        return output
    
    def evaluate_actions(self,state, next_states, dice_roll):
        action_values = np.zeros(4)
        actions = 0
        
        for i, next_state in enumerate(next_states):
            if next_state is False:
                action_values[i] = -1
            else:
                actions += 1

        if actions > 1:
            for i, next_state in enumerate(next_states):
                if next_state is False:
                    action_values[i] = -99999
                else:
                    AiInput = self.getGameState(next_state.state)
                    action_values[i] = GA.runModel(AiInput,GA.getPopulation(self.index))

        return np.argmax(action_values)

    def play(self,state, dice_roll, next_states):
        return self.evaluate_actions(state, next_states, dice_roll)
Q_DICT = dict()
if path.isfile("../QLearning/experience.csv"):
    experience = csv.reader(open("../QLearning/experience.csv"))
    for row in experience:
        if row:
            k,v = row
            v = np.fromstring(v[1:-1], sep=',').tolist()
            Q_DICT[k] = v

#agentB = GreedyQLearner(Q_DICT)

#pop = GAPopulation(GANNIndividual)
#pop.load_chromosomes(np.load("GANN/data/gen310.npy"))
#pop.evaulate_fitness_against_pop()

#agentB = GAIndividual()
#agentB.load_chromosome(np.load("GASimple/best_chromosomes/gen100.npy"))

#agentB = GANNIndividual()
#agentB.load_chromosome(np.load("GANN/best_chromosomes/gen310.npy"))

#agentB = SemiSmartPlayer()

#agentB = QSimple()
#agentB.actGreedy = True
#agentB.Q = np.loadtxt("../QLearning/QSimple/Q.txt")

#agentB = GAPlayer()

agentB = GAPlayer2()

#agentB = LudoPlayerRandom()

agentA = GAPlayer3()



savepath = "../Evaluation/GARAS3_vs_GARAS2.txt"
print("Will save at: "+savepath)

players = [ agentA, agentA, agentB, agentB ]

for i, player in enumerate(players):
    player.id = i
    print(player.name)

N = 1000
scores = []
for k in range(30):
    wins = [0, 0, 0, 0]
    for i in tqdm(range(N)):
        random.shuffle(players)
        ludoGame = LudoGame(players)
        winner = ludoGame.play_full_game()
        wins[players[winner].id] += 1
    print('Evaluation: '+str(k)+'/30')
    print("Wins: ",wins[1])
    scores.append(wins[1])

scores = np.array(scores)
np.savetxt(savepath, scores)



