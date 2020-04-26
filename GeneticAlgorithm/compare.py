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

#Q_DICT = dict()
#if path.isfile("../QLearning/experience.csv"):
#    experience = csv.reader(open("../QLearning/experience.csv"))
#    for row in experience:
#        if row:
#            k,v = row
#            v = np.fromstring(v[1:-1], sep=',').tolist()
#            Q_DICT[k] = v

#agentB = GreedyQLearner(Q_DICT)

#pop = GAPopulation(GANNIndividual)
#pop.load_chromosomes(np.load("GANN/data/gen310.npy"))
#pop.evaulate_fitness_against_pop()

#agentB = GAIndividual()
#agentB.load_chromosome(np.load("GASimple/best_chromosomes/gen100.npy"))

agentB = GANNIndividual()
agentB.load_chromosome(np.load("GANN/best_chromosomes/gen310.npy"))

agentA = QSimple()
agentA.actGreedy = True
agentA.Q = np.loadtxt("../QLearning/QSimple/Q.txt")

#agentB = SemiSmartPlayer()

savepath = "../Evaluation/QSimple_vs_GANN.txt"
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
    print(wins[1])
    scores.append(wins[1])

scores = np.array(scores)
np.savetxt(savepath, scores)



