import math

import numpy as np
import time

EPS = 1e-8

class MCTS(object):
    def __init__(self, game, policy_value_net, time_out, cpuct=1):
        self.game = game
        self.policy_value_net = policy_value_net
        self.time_out = time_out
        self.cpuct = cpuct
        self.Qsa = {}  # Q_value of s,a
        self.Nsa = {}  # n_visit_times of s,a
        self.Ns = {}  # n_visit_time of a board(s)
        self.Ps = {}  # policy by policy_value_net
        self.Es = {}  # game.getGameEnded of a board(s)
        self.Vs = {}  # game.getValidMoves of a board(s)

    def search(self, canonicalBoard):
        s = self.game.stringRepresentation(canonicalBoard)
        if s not in self.Es[s]:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]
        if s not in self.Ps[s]:
            self.Ps, v = self.policy_value_net.predict(canonicalBoard)
            self.Vs[s] = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * self.Vs[s]
            sum_Ps = np.sum(self.Ps[s])
            self.Ps[s] /= sum_Ps
            self.Ns[s] = 0
            return -v

        best_puct = -float('inf')
        best_a = -1
        for a in range(65):
            if self.Vs[s]:
                if (s, a) in self.Qsa:
                    puct = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    puct = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                if puct > best_puct:
                    best_puct = puct
                    best_a = a
        a = best_a
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        v = self.search(next_s)

        self.Ns[s] += 1
        if (s, a) in self.Qsa:
            self.Nsa[(s, a)] += 1
            self.Qsa[(s, a)] += (v - self.Qsa[(s, a)]) / self.Nsa[(s, a)]
        else:
            self.Nsa[(s, a)] = 1
            self.Qsa[(s, a)] = v
        return -v

    def getActionProb(self, canonicalBoard, temp=1):
        start_time = time.time()
        while time.time() - start_time < self.time_out:
            self.search(canonicalBoard)
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(65)]
        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * 65
            probs[best_a] = 1
            return probs
        counts = [x ** (1. / temp) for x in counts]
        sum_counts = float(sum(counts))
        probs = [x / sum_counts for x in counts]
        return probs
