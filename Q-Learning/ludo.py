import sys
sys.path.append('../../pyludo/')
from pyludo.utils import token_vulnerability, star_jump, will_send_self_home, will_send_opponent_home, is_globe_pos
from pyludo import LudoGame
import matplotlib.pyplot as plt
from os import path
from tqdm import tqdm
import numpy as np
import random
import time
import csv
    
Q_DICT = dict()

########################################################
#                SET TRAINING STATE                    #
########################################################
TRAINING = False                                       #
########################################################
if path.isfile("experience.csv"):
    experience = csv.reader(open("experience.csv"))
    for row in experience:
        if row:
            k,v = row
            v = np.fromstring(v[1:-1], sep=',').tolist()
            Q_DICT[k] = v

if TRAINING:
    trainingRewards = []
    if path.isfile("rewards.csv"):
        rewards = csv.reader(open("rewards.csv"))
        for row in rewards:
            if row:
                trainingRewards.append(float(row[0]))

agent = LudoQLearner(Q_DICT, training=TRAINING)
evaluator = LudoPlayerRandom()
players = [ agent, agent, evaluator, evaluator  ]

for i, player in enumerate(players):
    player.id = i

wins = [0, 0, 0, 0]

N = 1000

start_time = time.time()
for i in tqdm(range(N)):
    #print("Dict size: "+str(len(Q_DICT)))
    random.shuffle(players)
    
    if TRAINING:
        for p in players:
            if type(p) == LudoQLearner:
                trainingRewards.append(p.accumulated_reward)
                p.accumulated_reward = 0.0

    ludoGame = LudoGame(players)
    winner = ludoGame.play_full_game()
    wins[players[winner].id] += 1
    #if i % 100 == 0:
        #print('Game ', i, ' done')
duration = time.time() - start_time

if TRAINING:
    w = csv.writer(open("experience.csv", "w"))
    for key, val in Q_DICT.items():
        if val != [0] * 4:
            w.writerow([key, val])

    rewardFile = csv.writer(open("rewards.csv", "w"))
    for reward in trainingRewards:
        rewardFile.writerow([reward])

print('win distribution:', wins)
print('games per second:', N / duration)