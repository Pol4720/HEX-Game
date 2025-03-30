
import heapq

class HexAI:
    def __init__(self, board):
        self.board = board

    def a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def heuristic(self, node, goal):
        # Heurística admisible y consistente
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        return path[::-1]

    def get_neighbors(self, node):
        # Implementación para obtener vecinos de un nodo
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for direction in directions:
            neighbor = (node[0] + direction[0], node[1] + direction[1])
            if self.is_valid(neighbor):
                neighbors.append(neighbor)
        return neighbors

    def is_valid(self, node):
        # Verifica si un nodo es válido dentro del tablero
        x, y = node
        return 0 <= x < len(self.board) and 0 <= y < len(self.board[0]) and self.board[x][y] == 0

    def distance(self, node1, node2):
        # Distancia entre dos nodos
        return 1