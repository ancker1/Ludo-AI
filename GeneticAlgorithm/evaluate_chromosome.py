import sys
sys.path.append('../../pyludo/')
from pyludo import LudoGame
from static_players import LudoPlayerRandom
import random

def evaluate_agents(agents):
    players = agents
    ludoGame = LudoGame(players)
    return ludoGame.play_full_game()

def evaluate_agent(agent, amount_games):
    players = [ agent, LudoPlayerRandom(), LudoPlayerRandom(), LudoPlayerRandom() ]

    for i, player in enumerate(players):
        player.id = i

    wins = [0, 0, 0, 0]

    for i in range(amount_games):
        random.shuffle(players)
        ludoGame = LudoGame(players)
        winner = ludoGame.play_full_game()
        wins[players[winner].id] += 1
    return wins[0]