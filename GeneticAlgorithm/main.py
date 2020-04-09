import sys
sys.path.append('../../pyludo/')
from pyludo.utils import token_vulnerability, star_jump, will_send_self_home, will_send_opponent_home, is_globe_pos
from pyludo import LudoGame
from static_players import LudoPlayerRandom
from population import GAPopulation
from individual import GAIndividual
from selection import rank_selection
import random
import time
import numpy as np
from tqdm import tqdm

population = GAPopulation()
population.evaulate_fitness_against_pop()
best_chromosome = population.get_best_chromosome()
rank_selection(population, 10)


agent = GAIndividual()
agent.load_chromosome(best_chromosome)
print(best_chromosome)
players = [ agent, LudoPlayerRandom(), LudoPlayerRandom(), LudoPlayerRandom() ]

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
