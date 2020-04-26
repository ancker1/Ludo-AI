import sys
sys.path.append('../../pyludoperf/')
from pyludo import LudoGame
from players import LudoPlayerRandom
from QSimple import QSimple
from tqdm import tqdm
from os import path
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--movetype',     type=int,   default = 0)
args = parser.parse_args()


agent = QSimple(standardMoveType=args.movetype)
agent.actGreedy = True
agent.Q = np.loadtxt("QSimple/Q.txt")

evaluator = LudoPlayerRandom()
players = [ agent, agent, evaluator, evaluator ]

for i, player in enumerate(players):
    player.id = i

N = 1000
winrates = []
for k in range(30):
    wins = [0, 0, 0, 0]
    for i in tqdm(range(N)):
        random.shuffle(players)
        ludoGame = LudoGame(players)
        winner = ludoGame.play_full_game()
        wins[players[winner].id] += 1
    print(wins[1])
    winrates.append(wins[1])
np.savetxt("QSimple/winrates_movetype_"+str(args.movetype)+".txt", winrates)