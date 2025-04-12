import time
import math
import heapq
from hex_board import  HexBoard
from typing import  List

class EnhancedMCTSNode:
    def __init__(self, board: HexBoard, parent=None, move=None, player_id: int = 1):
        self.board = board.clone()
        self.parent = parent
        self.move = move
        self.children: List[EnhancedMCTSNode] = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = board.get_possible_moves()
        self.player_id = player_id  # 1 o 2 (🔴 o 🔵)

    def ucb1(self, exploration=1.4) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration * math.sqrt(math.log(self.parent.visits + 1e-6) / self.visits)

    def best_child(self) -> 'EnhancedMCTSNode':
        return max(self.children, key=lambda child: child.ucb1())

    def expand(self) -> 'EnhancedMCTSNode':
        move = self.select_heuristic_move()  # Usar heurística para expandir
        self.untried_moves.remove(move)
        new_board = self.board.clone()
        new_board.place_piece(*move, self.player_id)
        child = EnhancedMCTSNode(new_board, self, move, 3 - self.player_id)
        self.children.append(child)
        return child

    def is_terminal(self) -> bool:
        return self.board.check_connection(1) or self.board.check_connection(2) or not self.untried_moves

    def select_heuristic_move(self) -> tuple:
        # Priorizar movimientos con mejor valor heurístico
        moves = self.untried_moves
        move_scores = [(self._heuristic_score(move), move) for move in moves]
        move_scores.sort(reverse=True, key=lambda x: x[0])
        return move_scores[0][1]

    def _heuristic_score(self, move: tuple) -> float:
        # Crear un objeto HeuristicScore para calcular el puntaje
        heuristic_evaluator = HeuristicScore(self.player_id, 3 - self.player_id)
        
        # Clonar el tablero y aplicar el movimiento
        temp_board = self.board.clone()
        temp_board.place_piece(*move, self.player_id)
        
        # Calcular el puntaje heurístico combinado usando el objeto HeuristicScore
        score = heuristic_evaluator._combined_heuristic(temp_board)
        return score

    def _min_path_distance(self, board: HexBoard, player_id: int) -> float:
        # Implementación de Dijkstra para la distancia mínima entre lados
        size = board.size
        start_nodes = []
        target_side = set()
        
        if player_id == 1:  # Conectar izquierda (col 0) a derecha (col size-1)
            start_nodes = [(row, 0) for row in range(size) if board.board[row][0] == player_id]
            target_side = {(row, size-1) for row in range(size)}
        else:  # Conectar arriba (fila 0) a abajo (fila size-1)
            start_nodes = [(0, col) for col in range(size) if board.board[0][col] == player_id]
            target_side = {(size-1, col) for col in range(size)}
        
        # Dijkstra's algorithm
        visited = set()
        heap = []
        for node in start_nodes:
            heapq.heappush(heap, (0, node))
        
        while heap:
            dist, (r, c) = heapq.heappop(heap)
            if (r, c) in target_side:
                return dist
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for neighbor in self._get_neighbors(r, c):
                nr, nc = neighbor
                if board.board[nr][nc] == player_id or board.board[nr][nc] == 0:
                    heapq.heappush(heap, (dist + 1, neighbor))
        return float('inf')  # No hay camino

    def _bridge_control(self, board: HexBoard, move: tuple) -> int:
        # Verificar si el movimiento conecta dos grupos propios
        r, c = move
        groups = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.size and 0 <= nc < board.size:
                if board.board[nr][nc] == self.player_id:
                    groups += 1
        return 1 if groups >= 2 else 0

    def _get_neighbors(self, r: int, c: int) -> List[tuple]:
        # Implementación de adyacencias even-r
        neighbors = []
        
        dirs = [(0, -1),   # Izquierda
                (0, 1),    # Derecha
                (-1, 0),   # Arriba
                (1, 0),    # Abajo
                (-1, 1),   # Arriba derecha
                (1, -1)    ]
      
        
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.board.size and 0 <= nc < self.board.size:
                neighbors.append((nr, nc))
        return neighbors

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id   # Tu identificador (1 o 2)

    def play(self, board: HexBoard) -> tuple:
        raise NotImplementedError("¡Implementa este método!")


class AdvancedHexPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.time_limit = 10

    def play(self, board: HexBoard, time_limit: int = 9) -> tuple:
        self.time_limit = time_limit
        start_time = time.time()
        root = EnhancedMCTSNode(board, player_id=self.player_id)
        
        # Búsqueda MCTS con heurística
        iterations = 0
        while time.time() - start_time < self.time_limit - 0.1:  # Reservar 0.1s para selección
            node = self._select(root)
            if not node.is_terminal():
                node = node.expand()
            result = self._simulate(node)
            self._backpropagate(node, result)
            iterations += 1
        
        # Seleccionar el nodo con mejor ratio de victorias y visitas
        best_move = max(root.children, key=lambda child: (child.wins / child.visits) if child.visits > 0 else 0).move
        return best_move

    def _select(self, node: EnhancedMCTSNode) -> EnhancedMCTSNode:
        while node.children:
            node = node.best_child()
        return node

    def _simulate(self, node: EnhancedMCTSNode) -> float:
        # Simulación informada con heurística (no aleatoria)
        temp_board = node.board.clone()
        current_player = node.player_id
        steps = 0
        max_steps = temp_board.size * 2  # Limitar profundidad
        
        while steps < max_steps:
            if temp_board.check_connection(current_player):
                return 1.0 if current_player == self.player_id else 0.0
            moves = temp_board.get_possible_moves()
            # Elegir mejor movimiento según heurística
            move_scores = [(node._heuristic_score(move), move) for move in moves]
            best_move = max(move_scores, key=lambda x: x[0])[1]
            temp_board.place_piece(*best_move, current_player)
            current_player = 3 - current_player
            steps += 1
        return 0.5  # Empate por límite de pasos

    def _backpropagate(self, node: EnhancedMCTSNode, result: float):
        while node:
            node.visits += 1
            node.wins += result if node.player_id == self.player_id else (1 - result)
            node = node.parent



class HeuristicScore:
    def __init__(self, player_id: int, opponent_id: int, heuristic_weights=None):
        self.player_id = player_id
        self.opponent_id = opponent_id
        self.heuristic_cache = {}
        self.heuristic_weights = heuristic_weights or {
            'path_score': 0.3,
            'bridge_patterns': 0.2,
            'central_control': 0.2,
            'group_connectivity': 0.15,
            'potential_mobility': 0.1,
            'edge_proximity': 0.05
        }

    def _combined_heuristic(self, board: HexBoard) -> float:
        """Combina múltiples heurísticas avanzadas sin usar cache"""
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
        distance = [[math.inf] * size for _ in range(size)]
        heap = []
        
        if player_id == 1:
            for i in range(size):
                distance[i][0] = 0 if board.board[i][0] == 1 else 1
                heap.append((distance[i][0], i, 0))
        else:
            for j in range(size):
                distance[0][j] = 0 if board.board[0][j] == 2 else 1
                heap.append((distance[0][j], 0, j))
        
        heapq.heapify(heap)
        
        while heap:
            dist, i, j = heapq.heappop(heap)
            
            if (player_id == 1 and j == size - 1) or (player_id == 2 and i == size - 1):
                return dist
            
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    cost = 0 if board.board[ni][nj] == player_id else 1
                    new_dist = dist + cost
                    if new_dist < distance[ni][nj]:
                        distance[ni][nj] = new_dist
                        heapq.heappush(heap, (new_dist, ni, nj))
        
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
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
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
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
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

#sl23