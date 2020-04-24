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
    incrementer = 1

agent = typ()
evalplayer = LudoPlayerRandom()
players = [ agent, agent, evalplayer, evalplayer ]

for i, player in enumerate(players):
    player.id = i

N = 1000

current_generation = 0
while current_generation < generation_max:
    chromosome = np.load(pathprefix+"/best_chromosomes/gen{}.npy".format(str(current_generation)))
    agent.load_chromosome(chromosome)
    scores = []
    print('Evaluating gen: '+str(current_generation)+'/'+str(generation_max))
    for i in tqdm(range(10)):
        wins = [0, 0, 0, 0]
        for i in range(N):
            random.shuffle(players)
            ludoGame = LudoGame(players)
            winner = ludoGame.play_full_game()
            wins[players[winner].id] += 1
        scores.append(wins[1])
    np.savetxt(pathprefix+"/generational_winrates/gen{}.txt".format(str(current_generation)), scores)
    if current_generation == 20:
        incrementer = 5 # analyze all up to gen 20 - hereafter only every fifth.
    current_generation += incrementer

#scores = np.array(scores)
#np.savetxt(pathprefix+"/gen_vs_random.txt", scores)



