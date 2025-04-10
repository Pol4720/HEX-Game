import math
import time
from copy import deepcopy

ADJACENT_DIRECTIONS = [
    (0, -1), (0, 1), (-1, 0), (1, 0),
    (-1, 1), (1, -1)
]

class HexBoard:
    def __init__(self, size: int):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
    
    def clone(self) -> 'HexBoard':
        cloned = HexBoard(self.size)
        cloned.board = [row[:] for row in self.board]
        return cloned
    
    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        if self.board[row][col] == 0:
            self.board[row][col] = player_id
            return True
        return False
    
    def get_possible_moves(self) -> list:
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
    
    def check_connection(self, player_id: int) -> bool:
        visited = set()
        queue = []
        target = (self.size - 1) if player_id == 2 else None
        
        if player_id == 1:
            queue = [(i, 0) for i in range(self.size) if self.board[i][0] == 1]
            target_col = self.size - 1
        else:
            queue = [(0, j) for j in range(self.size) if self.board[0][j] == 2]
            target_row = self.size - 1

        for cell in queue:
            visited.add(cell)

        while queue:
            i, j = queue.pop(0)
            
            if (player_id == 1 and j == target_col) or (player_id == 2 and i == target_row):
                return True
            
            for di, dj in ADJACENT_DIRECTIONS:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    if self.board[ni][nj] == player_id and (ni, nj) not in visited:
                        visited.add((ni, nj))
                        queue.append((ni, nj))
        
        return False

class AlphaBetaHexPlayer:
    def __init__(self, player_id: int, timeout: float = 4.0, max_depth: int = 4):
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.timeout = timeout
        self.max_depth = max_depth
        self.start_time = 0
    
    def play(self, board: HexBoard) -> tuple:
        self.start_time = time.time()
        best_move = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        
        moves = board.get_possible_moves()
        moves.sort(key=lambda m: self.move_priority(m, board), reverse=True)
        
        for move in moves:
            if time.time() - self.start_time > self.timeout - 0.1:
                break
            
            new_board = board.clone()
            new_board.place_piece(*move, self.player_id)
            
            move_value = self.alphabeta(new_board, self.max_depth - 1, alpha, beta, False)
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
                alpha = max(alpha, best_value)
            
            if alpha >= beta:
                break
        
        return best_move or moves[0] if moves else (0, 0)
    
    def alphabeta(self, board: HexBoard, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        if time.time() - self.start_time > self.timeout - 0.05:
            return 0
        
        if board.check_connection(self.player_id):
            return math.inf
        if board.check_connection(self.opponent_id):
            return -math.inf
        
        if depth == 0:
            return self.evaluate(board)
        
        moves = board.get_possible_moves()
        
        if maximizing:
            value = -math.inf
            for move in moves:
                new_board = board.clone()
                new_board.place_piece(*move, self.player_id)
                value = max(value, self.alphabeta(new_board, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for move in moves:
                new_board = board.clone()
                new_board.place_piece(*move, self.opponent_id)
                value = min(value, self.alphabeta(new_board, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
    
    def evaluate(self, board: HexBoard) -> float:
        player_score = self.path_score(board, self.player_id)
        opponent_score = self.path_score(board, self.opponent_id)
        return opponent_score - player_score + self.control_score(board)
    
    def path_score(self, board: HexBoard, player_id: int) -> float:
        size = board.size
        distance_map = [[math.inf]*size for _ in range(size)]
        heap = []
        
        if player_id == 1:
            for i in range(size):
                distance_map[i][0] = 0 if board.board[i][0] == 1 else 1
                heap.append((distance_map[i][0], i, 0))
        else:
            for j in range(size):
                distance_map[0][j] = 0 if board.board[0][j] == 2 else 1
                heap.append((distance_map[0][j], 0, j))
        
        heap.sort()
        
        while heap:
            dist, i, j = heap.pop(0)
            
            if (player_id == 1 and j == size - 1) or (player_id == 2 and i == size - 1):
                return dist
            
            for di, dj in ADJACENT_DIRECTIONS:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    cost = 0 if board.board[ni][nj] == player_id else 1
                    new_dist = dist + cost
                    if new_dist < distance_map[ni][nj]:
                        distance_map[ni][nj] = new_dist
                        heap.append((new_dist, ni, nj))
                        heap.sort()
        
        return math.inf
    
    def control_score(self, board: HexBoard) -> float:
        size = board.size
        center = (size-1)/2
        score = 0
        for i in range(size):
            for j in range(size):
                if board.board[i][j] == self.player_id:
                    score += (1 - (abs(i-center) + abs(j-center))/size)
                elif board.board[i][j] == self.opponent_id:
                    score -= (1 - (abs(i-center) + abs(j-center))/size)
        return score * 2
    
    def move_priority(self, move: tuple, board: HexBoard) -> float:
        i, j = move
        priority = 0.0
        
        # Prefer moves near existing pieces
        for di, dj in ADJACENT_DIRECTIONS:
            ni, nj = i + di, j + dj
            if 0 <= ni < board.size and 0 <= nj < board.size:
                if board.board[ni][nj] == self.player_id:
                    priority += 2
                elif board.board[ni][nj] == self.opponent_id:
                    priority -= 1
        
        # Favor central positions
        center = (board.size - 1) / 2
        distance = abs(i - center) + abs(j - center)
        priority += (board.size - distance) * 0.5
        
        return priority

# class Player(AlphaBetaHexPlayer):
#     def __init__(self, player_id: int):
#         super().__init__(player_id, timeout=4.0, max_depth=4)