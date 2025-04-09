
class HexGame:
    def __init__(self, size):
        self.size = size
        self.board = self.create_board()

    def create_board(self):
        return [[None for _ in range(self.size)] for _ in range(self.size)]

    def make_move(self, player, position):
        if self.board[position[0]][position[1]] is None:
            self.board[position[0]][position[1]] = player
            return True
        return False