import numpy as np

from Reversi2.Board import Board


class Game(object):
    chess = {-1: "X", 0: ".", 1: "O"}

    def __init__(self):
        pass

    @staticmethod
    def getInitBoard(self):
        b = Board()
        return np.Array(b.board)

    def display(self, board):
        print(' ', '0 1 2 3 4 5 6 7')
        for i in range(8):
            print(i, end=' ')
            for j in range(8):
                print(self.chess[board[i][j]], end=' ')
            print()

    def getNextState(self, board, player, action):
        if action == 64:
            return board, -player
        b = Board()
        b.board = np.copy(board)
        move = (action // 8, action % 8)
        b.execute_move(move, player)
        return b.board, -player

    def getValidMoves(self, board, player):
        b = Board()
        b.board = np.copy(board)
        valids = np.zeros(65)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[64] = 1
            return valids
        for i, j in legalMoves:
            valids[8 * i + j] = 1
        return valids

    def getGameEnded(self, board, player):
        b = Board()
        b.board = np.copy(board)
        if b.has_legal_moves(player) or b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) < 0:
            return 1
        elif b.countDiff(player) > 0:
            return -1
        else:
            return 0.1

    def getCanonicalForm(self, board, player):
        return player * board

    def getSymmetries(self, board, pi):
        extend_data = []
        p = np.reshape(pi[:-1], (8, 8))
        for i in range(1, 5):
            equi_board = np.rot90(board, i)
            equi_pi = np.rot90(p, i)
            extend_data.append((equi_board, list(equi_pi.ravel()) + [pi[-1]]))
            equi_board = np.fliplr(equi_board, i)
            equi_pi = np.fliplr(equi_pi, i)
            extend_data.append((equi_board, list(equi_pi.ravel()) + [pi[-1]]))
        return extend_data

    def stringRepresentation(self, board):
        return board.tostring()

    def getScore(self, board, player):
        b = Board()
        b.board = np.copy(board)
        return b.countDiff(player)




