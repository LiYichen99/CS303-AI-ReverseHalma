import numpy as np
from Reversi.ReversiGame import ReversiGame as Game
import Arena
from MCTS import MCTS
from Reversi.ReversiPlayers import HumanReversiPlayer, RandomPlayer
from utils import *
from Reversi.Network.NNet import NNetWrapper


def test_game_with_human(game):
    hp = HumanReversiPlayer(game).play
    args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    nnet = NNetWrapper(game)
    nnet.load_checkpoint(folder='temp', filename='best.pth.tar')
    mcts = MCTS(game, nnet, args)
    n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

    arena = Arena.Arena(n1p, hp, game)
    print(arena.playGames(2, verbose=False))

def test_game_with_random(game):
    rp = RandomPlayer(game).play
    args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    nnet = NNetWrapper(game)
    nnet.load_checkpoint(folder='temp', filename='best.pth.tar')
    mcts = MCTS(game, nnet, args)
    n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

    arena = Arena.Arena(n1p, rp, game, game.display)
    print(arena.playGames(50, verbose=False))


if __name__ == '__main__':
    game = Game(8)
    test_game_with_random(game)
