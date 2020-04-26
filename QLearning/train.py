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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--movetype',     type=int,   default = 0)
args = parser.parse_args()

agent = QSimple(standardMoveType=args.movetype)
evaluator = LudoPlayerRandom()
players = [ agent, LudoPlayerRandom(), LudoPlayerRandom(), LudoPlayerRandom() ]

for i, player in enumerate(players):
    player.id = i

N = 500
rewards = []
wins = [0, 0, 0, 0]
for i in tqdm(range(N)):
    agent.moves = 0
    random.shuffle(players)
    ludoGame = LudoGame(players)
    winner = ludoGame.play_full_game()
    wins[players[winner].id] += 1
    rewards.append(agent.accumulatedReward)
    agent.accumulatedReward = 0

print(wins[0] / N)
np.savetxt("QSimple/Q.txt", agent.Q)
plt.plot(rewards, alpha=.2)
plt.plot(sg_filter.savitzky_golay(np.array(rewards), 31, 3))
plt.xlabel('Episodes', fontsize=16)
plt.ylabel('Accumulated reward', fontsize=16)
plt.grid()
plt.show()