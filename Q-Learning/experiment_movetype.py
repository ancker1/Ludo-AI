import sys
sys.path.append('../../pyludoperf/')
from pyludo import LudoGame
from players import LudoPlayerRandom
import matplotlib.pyplot as plt
from QSimple import QSimple
from tqdm import tqdm
from os import path
import numpy as np
import random


####################################
#   Train for 500 episodes         #
#   evaluate for 1000 episodes     #
####################################

for movetype in range(3):
    agent = QSimple(standardMoveType = movetype)
    players = [ agent, LudoPlayerRandom(), LudoPlayerRandom(), LudoPlayerRandom() ]

    for i, player in enumerate(players):
        player.id = i
    wins = [0] * 4
    N = 500
    for i in tqdm(range(N)):
        random.shuffle(players)
        ludoGame = LudoGame(players)
        winner = ludoGame.play_full_game()
        wins[players[winner].id] += 1
    print(wins)
    np.savetxt("QSimple/Q_movetype_"+str(movetype)+".txt", agent.Q)

    agent.actGreedy = True
    evaluator = LudoPlayerRandom()
    players = [ agent, agent, evaluator, evaluator ]
    for i, player in enumerate(players):
        player.id = i

    movetypewinrates = []
    for reps in tqdm(range(30)):
        wins = [0] * 4
        for i in range(N):
            random.shuffle(players)
            ludoGame = LudoGame(players)
            winner = ludoGame.play_full_game()
            wins[players[winner].id] += 1
        movetypewinrates.append(wins[1])
    np.savetxt("QSimple/winrates_movetype_"+str(movetype)+".txt", movetypewinrates)