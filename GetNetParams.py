from Reversi.ReversiGame import ReversiGame as Game
from Reversi.Network.NNet import NNetWrapper as nn
import numpy as np
import time

np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    g = Game(8)
    net = nn(g)
    net.load_checkpoint(folder="temp", filename="best.pth.tar")
    net_dict = {k: v.cpu().tolist() for k, v in net.nnet.state_dict().items()}
    print(net_dict.keys())
    with open("net_params.txt", 'w', encoding="utf8") as file:
        file.write(net_dict.__str__())
