import sys
sys.path.append('../../pyludo/')
from static_players import LudoPlayerRandom
from population import GAPopulation
from individual import GAIndividual
from pyludo import LudoGame
from tqdm import tqdm
import numpy as np
import random
import time


population = GAPopulation()
chromosomes = np.load("data/gen10.npy")
population.load_chromosomes(chromosomes)
population.evaulate_fitness_against_pop()
best_chromosome = population.get_best_chromosome()

agent = GAIndividual()
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