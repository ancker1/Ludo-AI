import sys
from tqdm import tqdm
from pyludo import LudoGame, LudoPlayerRandom
from GeneticAlgorithms import GeneticAlgorithmsClass

import random
import time
import numpy as np

def tanh(input):
    return np.tanh(input)

pops = 200
GA = GeneticAlgorithmsClass(58,0,0,1,tanh,pops)
#GA.importWeigths(0,"Gen-16-Wins-72-58-0-0-1-weigths.npy")


class GAPlayer:
    global GA
    """ takes a random valid action """
    name = 'Genetic algorithm'
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
                    action_values[i] = GA.runModel(AiInput,GA.getPopulation(self.index))

        return np.argmax(action_values)

    def play(self,state, dice_roll, next_states):
        return self.evaluate_actions(state, next_states, dice_roll)

def main():
    global GA
    GA1 = GAPlayer()
    GA2 = GAPlayer()
    GA3 = GAPlayer()
    GA4 = GAPlayer()
    
    randomAgent = LudoPlayerRandom()

    n = 50
    generations = 100
    for g in range(0,generations):
        print("Generation:",g)
        GA.exportPopulation()

        players = [GA1, GA2, GA3, GA4]
        for i, player in enumerate(players):
            player.id = i
        
        a = np.zeros(pops)
        a += n

        for _ in tqdm(range(int(n*pops*0.25))):
            try:
                idx = np.random.choice(pops,4,replace=False,p=np.divide(a,np.sum(a)))
            except:
                break

            a[idx] -= 1

            GA1.index = idx[0]
            GA2.index = idx[1]
            GA3.index = idx[2]
            GA4.index = idx[3]
            #print(a)
            ludoGame = LudoGame(players)
            winner = ludoGame.play_full_game()
            GA.populationScores[players[winner].index] += 1

        print(a)
        print(GA.populationScores)
        GA.populationScores = np.square(GA.populationScores)

        players = [GA1, randomAgent, randomAgent, randomAgent]
        for i, player in enumerate(players):
            player.id = i
        if True:
            score = [0, 0, 0, 0]

            print(np.argmax(GA.populationScores))
            GA1.index = np.argmax(GA.populationScores)

            for i in tqdm(range(1000)):
                random.shuffle(players)
                ludoGame = LudoGame(players)
                winner = ludoGame.play_full_game()
                score[players[winner].id] += 1
            
            GA.exportWeigths(GA1.index,score[0])

            print('win distribution:', score)
            f=open("output.txt", "a+")
            f.write(str(score[0])+"\n")
            f.close() 
        
        
        print("\t*Mutation*\t")
        GA.mutate()

if __name__ == "__main__":
    main()