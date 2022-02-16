import numpy as np


class Board(object):
    def __init__(self):
        self.board = np.zeros((8, 8))
        self.board[3][3] = 1
        self.board[3][4] = -1
        self.board[4][3] = -1
        self.board[4][4] = 1

    def countDiff(self, color):
        count = 0
        for y in range(8):
            for x in range(8):
                if self.board[x][y] == color:
                    count += 1
                if self.board[x][y] == -color:
                    count -= 1
        return count

    @staticmethod
    def is_on_board(x, y):
        return 0 <= x < 8 and 0 <= y < 8

    def is_legal_move(self, color, move):
        x_start, y_start = move
        if not (self.is_on_board(x_start, y_start) and self.board[x_start][y_start] == 0):
            return False
        for x_direction, y_direction in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x = x_start
            y = y_start
            x += x_direction
            y += y_direction
            if not self.is_on_board(x, y) or self.board[x][y] == 0 or self.board[x][y] == color:
                continue
            x += x_direction
            y += y_direction
            while self.is_on_board(x, y):
                if self.board[x][y] == 0:
                    break
                elif self.board[x][y] == color:
                    return True
                x += x_direction
                y += y_direction
        return False

    def get_legal_moves(self, color):
        moves = set()
        for i in range(8):
            for j in range(8):
                if self.is_valid_action(color, (i, j)):
                    moves.update((i, j))
        return moves

    def has_legal_moves(self, color):
        return len(self.get_legal_moves(color)) > 0

    def execute_move(self, move, color):
        x_start, y_start = move
        reverse_list = [move]
        for x_direction, y_direction in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x = x_start
            y = y_start
            x += x_direction
            y += y_direction
            if not self.is_on_board(x, y) or self.board[x][y] != -color:
                continue
            reverse_list.append((x, y))
            x += x_direction
            y += y_direction
            while self.is_on_board(x, y):
                if self.board[x][y] == 0:
                    break
                elif self.board[x][y] == -color:
                    reverse_list.append((x, y))
                    x += x_direction
                    y += y_direction
                else:
                    break
        assert len(reverse_list) > 0
        for i, j in reverse_list:
            self.board[i][j] = color



