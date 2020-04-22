import sys
sys.path.append('../../pyludoperf/')
from individual import GAIndividual, GANNIndividual
from fast_static_players import LudoPlayerRandom, SemiSmartPlayer
from population import GAPopulation
from pyludo import LudoGame
from tqdm import tqdm
import numpy as np
import argparse
import random
import time


parser = argparse.ArgumentParser()
parser.add_argument('--individual',     type=int,   default = 0     )
args = parser.parse_args()

if args.individual == 0:
    typ = GAIndividual
    generation_max = 101
    pathprefix = "GASimple"
    incrementer = 1
else:
    typ = GANNIndividual
    generation_max = 312
    pathprefix = "GANN"
    incrementer = 2

agent = typ()
evalplayer = LudoPlayerRandom()
players = [ agent, agent, evalplayer, evalplayer ]

for i, player in enumerate(players):
    player.id = i

N = 1000
scores = []
for k in range(1):
    for j in range(generation_max):
        chromosome = np.load("GASimple/best_chromosomes/gen{}.npy".format(str(j)))
        agent.load_chromosome(chromosome)
        wins = [0, 0, 0, 0]
        for i in tqdm(range(N)):
            random.shuffle(players)
            ludoGame = LudoGame(players)
            winner = ludoGame.play_full_game()
            wins[players[winner].id] += 1
        print('Chromosome: '+str(j)+'/100')
        scores.append(wins[1])

scores = np.array(scores)
np.savetxt(pathprefix+"/gen_vs_random.txt", scores)



