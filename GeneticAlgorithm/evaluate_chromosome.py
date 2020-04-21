import sys
sys.path.append('../../pyludoperf/')
from pyludo import LudoGame
from fast_static_players import LudoPlayerRandom
import random

def evaluate_agents(agents):
    players = agents
    ludoGame = LudoGame(players)
    return ludoGame.play_full_game()

def evaluate_agent(agent, amount_games):
    rndPlayer = LudoPlayerRandom()
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