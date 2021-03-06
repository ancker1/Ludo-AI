import sys
sys.path.append('../../pyludo/')
from static_players import LudoPlayerRandom
from population import GAPopulation
from individual import GAIndividual, GANNIndividual
from pyludo import LudoGame
from tqdm import tqdm
import numpy as np
import argparse
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('--individual',     type=int,   default = 0     )
parser.add_argument('--generation',     type=int,   default = 0     )
args = parser.parse_args()

if args.individual == 0:
    typ = GAIndividual
else:
    typ = GANNIndividual

population = GAPopulation(typ)
chromosomes = np.load("data/gen{}.npy".format(str(args.generation)))
population.load_chromosomes(chromosomes)
population.evaulate_fitness_against_pop()
best_chromosome = population.get_best_chromosome()

agent = typ()#GAIndividual()
agent.load_chromosome(best_chromosome)
print(best_chromosome)
players = [ agent, agent, LudoPlayerRandom(), LudoPlayerRandom() ]

for i, player in enumerate(players):
    player.id = i

wins = [0, 0, 0, 0]
N = 1000
start_time = time.time()
for i in tqdm(range(N)):
    random.shuffle(players)
    ludoGame = LudoGame(players)
    winner = ludoGame.play_full_game()
    wins[players[winner].id] += 1
duration = time.time() - start_time

print('win distribution:', wins)
print('games per second:', N / duration)