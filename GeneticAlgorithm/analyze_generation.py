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

typ = GAIndividual
population = GAPopulation(typ, population_size=100)

#current_generation = 0

#while current_generation < 312:
#chromosomes = np.load("data/gen{}.npy".format(str(current_generation)))
chromosomes = np.load("data/gen100.npy")
population.load_chromosomes(chromosomes)

population.evaluate_fitness_against_random()
#population.evaulate_fitness_against_pop()

scores = []

best_chromosome = population.get_best_chromosome()

agent = typ()
agent.load_chromosome(best_chromosome)

evalplayer = LudoPlayerRandom()
players = [ agent, agent, evalplayer, evalplayer ]

for i, player in enumerate(players):
    player.id = i

N = 1000

for j in range(30):
    wins = [0, 0, 0, 0]
    start_time = time.time()
    for i in tqdm(range(N)):
        random.shuffle(players)
        ludoGame = LudoGame(players)
        winner = ludoGame.play_full_game()
        wins[players[winner].id] += 1
    duration = time.time() - start_time
    print('Game: '+str(j)+'/30')
    print(wins[1])
    scores.append(wins[1])
    
#print('current generation:',current_generation)
print('win distribution:', wins)
print('games per second:', N / duration)

print(scores)
scores = np.array(scores)
np.savetxt("analyze/GASimple_vs_semismart.txt", scores)

#current_generation += 10


