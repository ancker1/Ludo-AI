import sys
sys.path.append('../../pyludoperf/')
from pyludo import LudoGame
from players import LudoPlayerRandom
import matplotlib.pyplot as plt
from QSimple import QSimple
from tqdm import tqdm
from os import path
import numpy as np
import sg_filter
import random
import csv

agent = QSimple()
agent.actGreedy = True
agent.Q = np.loadtxt("QSimple/Q.txt")

evaluator = LudoPlayerRandom()
players = [ agent, agent, evaluator, evaluator ]

for i, player in enumerate(players):
    player.id = i

N = 1000
rewards = []
wins = [0, 0, 0, 0]
for i in tqdm(range(N)):
    random.shuffle(players)
    ludoGame = LudoGame(players)
    winner = ludoGame.play_full_game()
    wins[players[winner].id] += 1
print(wins)