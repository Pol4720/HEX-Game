import math
import time
from copy import deepcopy


ADJACENT_DIRECTIONS = [
    (0, -1),   # Izquierda
    (0, 1),    # Derecha
    (-1, 0),   # Arriba
    (1, 0),    # Abajo
    (-1, 1),   # Arriba derecha
    (1, -1)    # Abajo izquierda
]

class HexBoard:
    def __init__(self, size: int):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
    
    def clone(self) -> 'HexBoard':
        """Devuelve una copia profunda del tablero actual"""
        new_board = HexBoard(self.size)
        new_board.board = deepcopy(self.board)
        return new_board
    
    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        """Coloca una ficha si la casilla está vacía."""
        if self.board[row][col] == 0:
            self.board[row][col] = player_id
            return True
        return False
    
    def get_possible_moves(self) -> list:
        """Devuelve todas las casillas vacías como tuplas (fila, columna)."""
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves
    
    def check_connection(self, player_id: int) -> bool:
        """Verifica si el jugador ha conectado sus dos lados usando BFS."""
        visited = set()
        queue = []
        
        # Jugador 1 (🔴) necesita conectar izquierda (col=0) a derecha (col=size-1)
        # Jugador 2 (🔵) necesita conectar arriba (fila=0) a abajo (fila=size-1)
        
        if player_id == 1:
            # Inicializar cola con todas las celdas del lado izquierdo
            for i in range(self.size):
                if self.board[i][0] == player_id:
                    queue.append((i, 0))
                    visited.add((i, 0))
        else:
            # Inicializar cola con todas las celdas del lado superior
            for j in range(self.size):
                if self.board[0][j] == player_id:
                    queue.append((0, j))
                    visited.add((0, j))
        
        # BFS para encontrar conexión al otro lado
        while queue:
            i, j = queue.pop(0)
            
            # Verificar si hemos llegado al otro lado
            if player_id == 1 and j == self.size - 1:
                return True
            if player_id == 2 and i == self.size - 1:
                return True
            
            # Explorar vecinos
            for di, dj in ADJACENT_DIRECTIONS:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    if self.board[ni][nj] == player_id and (ni, nj) not in visited:
                        visited.add((ni, nj))
                        queue.append((ni, nj))
        
        return False

class AIPlayer:
    def __init__(self, player_id: int, max_depth: int = 3, timeout: float = 5.0):
        self.player_id = player_id
        self.opponent_id = 3 - player_id  # Calcula el ID del oponente (1->2 o 2->1)
        self.max_depth = max_depth
        self.timeout = timeout
        self.start_time = 0
    
    def play(self, board: HexBoard) -> tuple:
        """Decide la mejor jugada usando Minimax con poda Alpha-Beta."""
        self.start_time = time.time()
        best_move = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        
        # Ordenar movimientos por heurística para mejorar la poda
        possible_moves = board.get_possible_moves()
        possible_moves.sort(key=lambda move: self.move_heuristic(board, move, self.player_id), reverse=True)
        
        for move in possible_moves:
            if time.time() - self.start_time > self.timeout - 0.1:  # Dejar un margen de seguridad
                break
                
            new_board = board.clone()
            new_board.place_piece(move[0], move[1], self.player_id)
            
            # Llamada recursiva a minimax
            value = self.minimax(new_board, self.max_depth - 1, alpha, beta, False)
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break  # Poda beta
        
        # Si no encontramos mejor movimiento (por tiempo), elegir el primero disponible
        return best_move if best_move else possible_moves[0] if possible_moves else (0, 0)
    
    def minimax(self, board: HexBoard, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """Implementación de Minimax con poda Alpha-Beta."""
        # Verificar tiempo límite
        if time.time() - self.start_time > self.timeout - 0.05:
            return 0
        
        # Verificar si el juego ha terminado
        if board.check_connection(self.player_id):
            return math.inf
        if board.check_connection(self.opponent_id):
            return -math.inf
        
        # Si alcanzamos la profundidad máxima, usar función de evaluación
        if depth == 0:
            return self.evaluate_board(board)
        
        possible_moves = board.get_possible_moves()
        
        if is_maximizing:
            value = -math.inf
            for move in possible_moves:
                new_board = board.clone()
                new_board.place_piece(move[0], move[1], self.player_id)
                value = max(value, self.minimax(new_board, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Poda beta
            return value
        else:
            value = math.inf
            for move in possible_moves:
                new_board = board.clone()
                new_board.place_piece(move[0], move[1], self.opponent_id)
                value = min(value, self.minimax(new_board, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break  # Poda alpha
            return value
    
    def evaluate_board(self, board: HexBoard) -> float:
        """Función de evaluación heurística para el estado del tablero."""
        # Puntuación base
        score = 0
        
        # 1. Conexiones potenciales
        player_path = self.shortest_path_length(board, self.player_id)
        opponent_path = self.shortest_path_length(board, self.opponent_id)
        
        # La diferencia en las longitudes de camino más corto es importante
        if opponent_path == 0:  # El oponente ya ganó
            return -math.inf
        if player_path == 0:  # Ya ganamos
            return math.inf
        
        score += (1.0 / player_path - 1.0 / opponent_path) * 100
        
        # 2. Control del centro (las celdas centrales son más valiosas)
        center_weight = 0
        center = board.size // 2
        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] == self.player_id:
                    # Distancia al centro (cuanto más cerca, mejor)
                    distance = abs(i - center) + abs(j - center)
                    center_weight += (board.size - distance)
                elif board.board[i][j] == self.opponent_id:
                    distance = abs(i - center) + abs(j - center)
                    center_weight -= (board.size - distance)
        score += center_weight * 0.5
        
        # 3. Grupos conectados (cuanto más grandes mejor)
        player_groups = self.find_connected_groups(board, self.player_id)
        opponent_groups = self.find_connected_groups(board, self.opponent_id)
        
        max_player_group = max(len(group) for group in player_groups) if player_groups else 0
        max_opponent_group = max(len(group) for group in opponent_groups) if opponent_groups else 0
        
        score += (max_player_group - max_opponent_group) * 2
        
        return score
    
    def shortest_path_length(self, board: HexBoard, player_id: int) -> int:
        """Calcula la longitud del camino más corto para conectar los lados usando Dijkstra."""
        size = board.size
        distances = {}
        heap = []
        
        # Inicializar distancias para el jugador 1 (izquierda a derecha)
        if player_id == 1:
            for i in range(size):
                if board.board[i][0] == player_id:
                    distances[(i, 0)] = 0
                    heap.append((0, i, 0))
                else:
                    distances[(i, 0)] = 1 if board.board[i][0] == 0 else math.inf
                    heap.append((distances[(i, 0)], i, 0))
            
            # Inicializar el resto de celdas
            for i in range(size):
                for j in range(1, size):
                    distances[(i, j)] = math.inf
        
        # Inicializar distancias para el jugador 2 (arriba a abajo)
        else:
            for j in range(size):
                if board.board[0][j] == player_id:
                    distances[(0, j)] = 0
                    heap.append((0, 0, j))
                else:
                    distances[(0, j)] = 1 if board.board[0][j] == 0 else math.inf
                    heap.append((distances[(0, j)], 0, j))
            
            # Inicializar el resto de celdas
            for i in range(1, size):
                for j in range(size):
                    distances[(i, j)] = math.inf
        
        # Ordenar el heap
        heap.sort()
        
        # Dijkstra
        while heap:
            current_dist, i, j = heap.pop(0)
            
            # Si ya encontramos un camino al otro lado, retornar
            if player_id == 1 and j == size - 1:
                return current_dist
            if player_id == 2 and i == size - 1:
                return current_dist
            
            if current_dist > distances[(i, j)]:
                continue
            
            # Explorar vecinos
            for di, dj in ADJACENT_DIRECTIONS:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    # Costo de mover a la celda vecina
                    if board.board[ni][nj] == player_id:
                        cost = 0
                    elif board.board[ni][nj] == 0:
                        cost = 1
                    else:
                        cost = math.inf  # Celda del oponente, no se puede pasar
                    
                    if distances[(ni, nj)] > current_dist + cost:
                        distances[(ni, nj)] = current_dist + cost
                        heap.append((distances[(ni, nj)], ni, nj))
                        heap.sort()
        
        # Si no hay camino, retornar infinito
        return math.inf
    
    def find_connected_groups(self, board: HexBoard, player_id: int) -> list:
        """Encuentra todos los grupos conectados de fichas del jugador."""
        size = board.size
        visited = set()
        groups = []
        
        for i in range(size):
            for j in range(size):
                if board.board[i][j] == player_id and (i, j) not in visited:
                    # BFS para encontrar el grupo conectado
                    queue = [(i, j)]
                    visited.add((i, j))
                    group = [(i, j)]
                    
                    while queue:
                        ci, cj = queue.pop(0)
                        for di, dj in ADJACENT_DIRECTIONS:
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < size and 0 <= nj < size:
                                if board.board[ni][nj] == player_id and (ni, nj) not in visited:
                                    visited.add((ni, nj))
                                    group.append((ni, nj))
                                    queue.append((ni, nj))
                    
                    groups.append(group)
        
        return groups
    
    def move_heuristic(self, board: HexBoard, move: tuple, player_id: int) -> float:
        """Heurística rápida para ordenar movimientos potenciales."""
        i, j = move
        score = 0
        
        # 1. Preferir celdas que conecten con nuestras propias fichas
        for di, dj in ADJACENT_DIRECTIONS:
            ni, nj = i + di, j + dj
            if 0 <= ni < board.size and 0 <= nj < board.size:
                if board.board[ni][nj] == player_id:
                    score += 2
                elif board.board[ni][nj] == 3 - player_id:
                    score -= 1
        
        # 2. Preferir celdas centrales
        center = board.size / 2
        distance_to_center = abs(i - center) + abs(j - center)
        score += (board.size - distance_to_center) * 0.5
        
        # 3. Preferir celdas que bloqueen al oponente
        # Simular el movimiento y ver si bloquea un camino del oponente
        temp_board = board.clone()
        temp_board.place_piece(i, j, player_id)
        opponent_path_before = self.shortest_path_length(board, 3 - player_id)
        opponent_path_after = self.shortest_path_length(temp_board, 3 - player_id)
        if opponent_path_after > opponent_path_before:
            score += (opponent_path_after - opponent_path_before) * 0.5
        
        return score
    
# class Player:
#     def __init__(self, player_id: int):
#         self.player_id = player_id  # Tu identificador (1 o 2)

#     def play(self, board: HexBoard) -> tuple:
#         raise NotImplementedError("¡Implementa este método!")

# # Clase final que hereda de Player para cumplir con la interfaz requerida
# class Player(AIPlayer):
#     def __init__(self, player_id: int):
#         super().__init__(player_id, max_depth=3, timeout=2.0)
    
#     def play(self, board: HexBoard) -> tuple:
#         return super().play(board)