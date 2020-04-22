import sys
sys.path.append('../../pyludoperf/')
from pyludo import LudoGame
from fast_static_players import LudoPlayerRandom, SemiSmartPlayer
import random

def evaluate_agents(agents):
    players = agents
    ludoGame = LudoGame(players)
    return ludoGame.play_full_game()

def evaluate_agent(agent, amount_games, use_random = True):
    if use_random:
        rndPlayer = LudoPlayerRandom()
    else:
        rndPlayer = SemiSmartPlayer()
    players = [ agent, agent, rndPlayer, rndPlayer ]

    for i, player in enumerate(players):
        player.id = i

    wins = [0, 0, 0, 0]

    for i in range(amount_games):
        random.shuffle(players)
        ludoGame = LudoGame(players)
        winner = ludoGame.play_full_game()
        wins[players[winner].id] += 1
    return wins[1]