import sys
from tqdm import tqdm
from pyludo import LudoGame, LudoPlayerRandom
from GeneticAlgorithms import GeneticAlgorithmsClass

import random
import time
import numpy as np

def tanh(input):
    return np.tanh(input)

pops = 100
GA = GeneticAlgorithmsClass(60,0,0,1,tanh,pops)
#GA.importWeigths(0,"Gen-76-Wins-904-60-0-0-1-weigths.npy")
GA.importWeigths(0,"Gen-1-Wins-910-60-0-0-1-weigths.npy")


class GAPlayer:
    global GA
    name = 'Genetic algorithm'
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
                    action_values[i] = GA.runModel(AiInput,GA.getPopulation(self.index))

        return np.argmax(action_values)

    def play(self,state, dice_roll, next_states):
        return self.evaluate_actions(state, next_states, dice_roll)


def main():
    global GA
    GA1 = GAPlayer()
    GA2 = GAPlayer()

    randomAgent = LudoPlayerRandom()
    print(GA.getPopulation(0))

    for t in range(0,100):
        print("Test:",t)
        players = [GA1, GA1, GA2, GA2]
        for i, player in enumerate(players):
            player.id = i
        GA1.index = 0
        GA2.index = 1

        score = [0,0,0,0]
        for _ in tqdm(range(1000)):
            random.shuffle(players)
            ludoGame = LudoGame(players)
            winner = ludoGame.play_full_game()
            score[players[winner].id] += 1
        print("Wins: ",score)


if __name__ == "__main__":
    main()