import sys
sys.path.append('../../pyludoperf/')
from pyludo import LudoGame
from players import GreedyQLearner, LudoPlayerRandom, SemiSmartPlayer
from .GeneticAlgorithm.individual import GAIndividual
from tqdm import tqdm
from os import path
import numpy as np
import random
import csv

Q_DICT = dict()
if path.isfile("experience.csv"):
    experience = csv.reader(open("experience.csv"))
    for row in experience:
        if row:
            k,v = row
            v = np.fromstring(v[1:-1], sep=',').tolist()
            Q_DICT[k] = v

agent = GreedyQLearner(Q_DICT)
evaluator = SemiSmartPlayer()
players = [ agent, agent, evaluator, evaluator  ]

for i, player in enumerate(players):
    player.id = i



N = 1000
scores = []
for j in range(30):
    print("Game: "+str(j)+"/30")
    wins = [0, 0, 0, 0]
    for i in tqdm(range(N)):
        random.shuffle(players)
        ludoGame = LudoGame(players)
        winner = ludoGame.play_full_game()
        wins[players[winner].id] += 1
    print(wins)
    scores.append(wins[1])
print(scores)
scores = np.array(scores)
np.savetxt("../Evaluation/QLearner_vs_semismart.txt", scores)