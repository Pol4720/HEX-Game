import math
import time
import pickle
import os
from collections import defaultdict
from copy import deepcopy

ADJACENT_DIRECTIONS = [
    (0, -1), (0, 1), (-1, 0), (1, 0),
    (-1, 1), (1, -1)
]

# Configuración avanzada
CACHE_FILE = "hex_heuristic_cache.pkl"
OPENING_BOOK_FILE = "hex_opening_book.pkl"
MEMORY_LIMIT_MB = 500  # Límite de memoria para cache

class HexBoard:
    def __init__(self, size: int):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.zobrist_hash = 0
        self._init_zobrist()
    
    def _init_zobrist(self):
        """Inicializa tablas Zobrist para hashing eficiente"""
        self.zobrist_table = [[[0]*3 for _ in range(self.size)] for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                for k in range(3):
                    self.zobrist_table[i][j][k] = hash(f"{i},{j},{k}") & 0xFFFFFFFF
    
    def clone(self) -> 'HexBoard':
        cloned = HexBoard(self.size)
        cloned.board = [row[:] for row in self.board]
        cloned.zobrist_hash = self.zobrist_hash
        return cloned
    
    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        if self.board[row][col] == 0:
            # Actualizar hash Zobrist
            self.zobrist_hash ^= self.zobrist_table[row][col][0]  # Quitar vacío
            self.zobrist_hash ^= self.zobrist_table[row][col][player_id]  # Añadir jugador
            self.board[row][col] = player_id
            return True
        return False
    
    def get_possible_moves(self) -> list:
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
    
    def check_connection(self, player_id: int) -> bool:
        """Versión optimizada con memoización parcial"""
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

class UltraHexPlayer:
    def __init__(self, player_id: int, timeout: float = 5.0, max_depth: int = 6):
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.timeout = timeout
        self.max_depth = max_depth
        self.start_time = 0
        self.transposition_table = {}
        self.opening_book = self._load_opening_book()
        self.heuristic_cache = self._load_heuristic_cache()
        self.heuristic_weights = {
            'path_score': 1.5,
            'bridge_patterns': 1.2,
            'central_control': 0.8,
            'group_connectivity': 1.0,
            'potential_mobility': 0.7,
            'edge_proximity': 0.5
        }
    
    def _load_opening_book(self):
        """Carga la base de datos de aperturas desde archivo"""
        if os.path.exists(OPENING_BOOK_FILE):
            try:
                with open(OPENING_BOOK_FILE, 'rb') as f:
                    return pickle.load(f)
            except:
                return defaultdict(list)
        return defaultdict(list)
    
    def _load_heuristic_cache(self):
        """Carga la cache de heurísticas con límite de memoria"""
        cache = {}
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'rb') as f:
                    cache = pickle.load(f)
                    # Limitar tamaño de cache
                    if len(cache) > MEMORY_LIMIT_MB * 1024 * 1024 / 100:  # ~100 bytes por entrada
                        cache.clear()
            except:
                pass
        return cache
    
    def _save_cache(self):
        """Guarda la cache en disco"""
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(self.heuristic_cache, f)
        except:
            pass
    
    def play(self, board: HexBoard) -> tuple:
        self.start_time = time.time()
        
        # Primero verificar si estamos en una apertura conocida
        opening_move = self._check_opening_book(board)
        if opening_move:
            return opening_move
        
        # Búsqueda principal con múltiples heurísticas
        best_move = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        
        moves = board.get_possible_moves()
        moves.sort(key=lambda m: self._combined_move_priority(m, board), reverse=True)
        
        for move in moves:
            if time.time() - self.start_time > self.timeout - 0.1:
                break
            
            new_board = board.clone()
            new_board.place_piece(*move, self.player_id)
            
            move_value = self._enhanced_alphabeta(new_board, self.max_depth - 1, alpha, beta, False)
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
                alpha = max(alpha, best_value)
            
            if alpha >= beta:
                break
        
        # Guardar cache periódicamente
        if len(self.heuristic_cache) % 100 == 0:
            self._save_cache()
        
        return best_move or moves[0] if moves else (0, 0)
    
    def _check_opening_book(self, board: HexBoard) -> tuple:
        """Consulta la base de datos de aperturas"""
        board_hash = board.zobrist_hash
        if board_hash in self.opening_book:
            for move, score in self.opening_book[board_hash]:
                if board.board[move[0]][move[1]] == 0:  # Verificar que el movimiento es válido
                    return move
        return None
    
    def _enhanced_alphabeta(self, board: HexBoard, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """Versión mejorada de Alpha-Beta con múltiples optimizaciones"""
        if time.time() - self.start_time > self.timeout - 0.05:
            return 0
        
        # Verificar transposition table
        board_hash = board.zobrist_hash
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            if entry['depth'] >= depth:
                if entry['flag'] == 'EXACT':
                    return entry['value']
                elif entry['flag'] == 'LOWERBOUND':
                    alpha = max(alpha, entry['value'])
                elif entry['flag'] == 'UPPERBOUND':
                    beta = min(beta, entry['value'])
                if alpha >= beta:
                    return entry['value']
        
        # Condiciones terminales
        if board.check_connection(self.player_id):
            return math.inf
        if board.check_connection(self.opponent_id):
            return -math.inf
        
        if depth == 0:
            return self._combined_heuristic(board)
        
        moves = board.get_possible_moves()
        if not moves:
            return 0
        
        # Ordenar movimientos basado en heurísticas simples
        moves.sort(key=lambda m: self._combined_move_priority(m, board), reverse=maximizing)
        
        best_value = -math.inf if maximizing else math.inf
        best_move = None
        
        for move in moves:
            new_board = board.clone()
            new_board.place_piece(*move, self.player_id if maximizing else self.opponent_id)
            
            value = self._enhanced_alphabeta(new_board, depth - 1, alpha, beta, not maximizing)
            
            if maximizing:
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
            
            if alpha >= beta:
                break
        
        # Guardar en transposition table
        entry = {
            'value': best_value,
            'depth': depth,
            'flag': 'EXACT' if best_value <= alpha else ('LOWERBOUND' if best_value >= beta else 'UPPERBOUND'),
            'best_move': best_move
        }
        self.transposition_table[board_hash] = entry
        
        return best_value
    
    def _combined_heuristic(self, board: HexBoard) -> float:
        """Combina múltiples heurísticas avanzadas"""
        cache_key = board.zobrist_hash
        if cache_key in self.heuristic_cache:
            return self.heuristic_cache[cache_key]
        
        # Calcular todas las heurísticas
        heuristics = {
            'path_score': self._path_heuristic(board),
            'bridge_patterns': self._bridge_pattern_heuristic(board),
            'central_control': self._central_control_heuristic(board),
            'group_connectivity': self._group_connectivity_heuristic(board),
            'potential_mobility': self._potential_mobility_heuristic(board),
            'edge_proximity': self._edge_proximity_heuristic(board)
        }
        
        # Combinar con pesos ajustables
        total = sum(heuristics[h] * self.heuristic_weights[h] for h in heuristics)
        normalized = total / sum(self.heuristic_weights.values())
        
        # Almacenar en cache
        self.heuristic_cache[cache_key] = normalized
        
        return normalized
    
    def _path_heuristic(self, board: HexBoard) -> float:
        """Heurística basada en caminos más cortos"""
        player_path = self._shortest_path(board, self.player_id)
        opponent_path = self._shortest_path(board, self.opponent_id)
        
        if player_path == 0:
            return math.inf
        if opponent_path == 0:
            return -math.inf
        
        return (1.0 / (player_path + 0.1)) - (1.0 / (opponent_path + 0.1))
    
    def _shortest_path(self, board: HexBoard, player_id: int) -> float:
        """Versión optimizada del cálculo de camino más corto"""
        size = board.size
        distance = [[math.inf]*size for _ in range(size)]
        heap = []
        
        if player_id == 1:
            for i in range(size):
                distance[i][0] = 0 if board.board[i][0] == 1 else 1
                heap.append((distance[i][0], i, 0))
        else:
            for j in range(size):
                distance[0][j] = 0 if board.board[0][j] == 2 else 1
                heap.append((distance[0][j], 0, j))
        
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
                    if new_dist < distance[ni][nj]:
                        distance[ni][nj] = new_dist
                        heap.append((new_dist, ni, nj))
                        heap.sort()
        
        return math.inf
    
    def _bridge_pattern_heuristic(self, board: HexBoard) -> float:
        """Detecta y puntúa patrones de puente (estrategia clave en HEX)"""
        score = 0
        size = board.size
        
        for i in range(size - 1):
            for j in range(size - 1):
                # Patrón de puente clásico
                if (board.board[i][j] == self.player_id and 
                    board.board[i+1][j+1] == self.player_id and 
                    board.board[i][j+1] == 0 and 
                    board.board[i+1][j] == 0):
                    score += 0.3
                
                # Patrón de puente invertido
                if (board.board[i][j+1] == self.player_id and 
                    board.board[i+1][j] == self.player_id and 
                    board.board[i][j] == 0 and 
                    board.board[i+1][j+1] == 0):
                    score += 0.3
                
                # Versiones para el oponente (penalizar)
                if (board.board[i][j] == self.opponent_id and 
                    board.board[i+1][j+1] == self.opponent_id and 
                    board.board[i][j+1] == 0 and 
                    board.board[i+1][j] == 0):
                    score -= 0.2
                
                if (board.board[i][j+1] == self.opponent_id and 
                    board.board[i+1][j] == self.opponent_id and 
                    board.board[i][j] == 0 and 
                    board.board[i+1][j+1] == 0):
                    score -= 0.2
        
        return score / (size * 0.5)  # Normalizar por tamaño del tablero
    
    def _central_control_heuristic(self, board: HexBoard) -> float:
        """Evalúa el control del centro del tablero"""
        size = board.size
        center = (size - 1) / 2
        score = 0
        
        for i in range(size):
            for j in range(size):
                distance = math.sqrt((i - center)**2 + (j - center)**2)
                weight = 1 - (distance / (center * math.sqrt(2)))
                
                if board.board[i][j] == self.player_id:
                    score += weight
                elif board.board[i][j] == self.opponent_id:
                    score -= weight
        
        return score / (size * 0.5)  # Normalizar
    
    def _group_connectivity_heuristic(self, board: HexBoard) -> float:
        """Evalúa la conectividad de grupos de fichas"""
        player_groups = self._find_connected_groups(board, self.player_id)
        opponent_groups = self._find_connected_groups(board, self.opponent_id)
        
        player_score = sum(len(g)**1.5 for g in player_groups)
        opponent_score = sum(len(g)**1.5 for g in opponent_groups)
        
        return (player_score - opponent_score) / (board.size**2 * 2)
    
    def _find_connected_groups(self, board: HexBoard, player_id: int) -> list:
        """Encuentra grupos conectados usando BFS"""
        size = board.size
        visited = set()
        groups = []
        
        for i in range(size):
            for j in range(size):
                if board.board[i][j] == player_id and (i, j) not in visited:
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
    
    def _potential_mobility_heuristic(self, board: HexBoard) -> float:
        """Evalúa la movilidad potencial (espacios vacíos adyacentes a nuestras fichas)"""
        player_mobility = 0
        opponent_mobility = 0
        size = board.size
        
        for i in range(size):
            for j in range(size):
                if board.board[i][j] == 0:
                    for di, dj in ADJACENT_DIRECTIONS:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if board.board[ni][nj] == self.player_id:
                                player_mobility += 1
                                break
                            elif board.board[ni][nj] == self.opponent_id:
                                opponent_mobility += 1
                                break
        
        return (player_mobility - opponent_mobility) / (size * size * 2)
    
    def _edge_proximity_heuristic(self, board: HexBoard) -> float:
        """Evalúa la proximidad estratégica a los bordes relevantes"""
        score = 0
        size = board.size
        
        if self.player_id == 1:  # Jugador horizontal (izquierda-derecha)
            for i in range(size):
                for j in [0, size-1]:  # Bordes izquierdo y derecho
                    if board.board[i][j] == self.player_id:
                        score += 0.5 if j == 0 else 0.5
                    elif board.board[i][j] == self.opponent_id:
                        score -= 0.3 if j == 0 else 0.3
        else:  # Jugador vertical (arriba-abajo)
            for j in range(size):
                for i in [0, size-1]:  # Bordes superior e inferior
                    if board.board[i][j] == self.player_id:
                        score += 0.5 if i == 0 else 0.5
                    elif board.board[i][j] == self.opponent_id:
                        score -= 0.3 if i == 0 else 0.3
        
        return score / size
    
    def _combined_move_priority(self, move: tuple, board: HexBoard) -> float:
        """Prioridad combinada para ordenar movimientos"""
        i, j = move
        priority = 0.0
        
        # 1. Proximidad a fichas existentes
        for di, dj in ADJACENT_DIRECTIONS:
            ni, nj = i + di, j + dj
            if 0 <= ni < board.size and 0 <= nj < board.size:
                if board.board[ni][nj] == self.player_id:
                    priority += 2.0
                elif board.board[ni][nj] == self.opponent_id:
                    priority -= 1.0
        
        # 2. Control central
        center = (board.size - 1) / 2
        distance = math.sqrt((i - center)**2 + (j - center)**2)
        priority += (board.size - distance) * 0.7
        
        # 3. Proximidad a bordes estratégicos
        if self.player_id == 1:  # Horizontal
            edge_dist = min(j, board.size - 1 - j)
        else:  # Vertical
            edge_dist = min(i, board.size - 1 - i)
        priority += (board.size - edge_dist) * 0.4
        
        # 4. Patrones de puente potenciales
        priority += self._bridge_move_potential(move, board) * 1.5
        
        return priority
    
    def _bridge_move_potential(self, move: tuple, board: HexBoard) -> float:
        """Evalúa potencial para crear puentes"""
        i, j = move
        score = 0
        size = board.size
        
        # Verificar si este movimiento completa un puente
        for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni1, nj1 = i + di, j
            ni2, nj2 = i, j + dj
            if (0 <= ni1 < size and 0 <= nj1 < size and
                0 <= ni2 < size and 0 <= nj2 < size):
                if (board.board[ni1][nj1] == self.player_id and
                    board.board[ni2][nj2] == self.player_id):
                    score += 0.4
        
        # Verificar si bloquea un puente del oponente
        for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni1, nj1 = i + di, j
            ni2, nj2 = i, j + dj
            if (0 <= ni1 < size and 0 <= nj1 < size and
                0 <= ni2 < size and 0 <= nj2 < size):
                if (board.board[ni1][nj1] == self.opponent_id and
                    board.board[ni2][nj2] == self.opponent_id):
                    score += 0.3
        
        return score

# class Player(UltraHexPlayer):
#     def __init__(self, player_id: int):
#         super().__init__(player_id, timeout=4.0, max_depth=4)