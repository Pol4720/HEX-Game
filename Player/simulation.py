from player import  HexBoard
from MCplayer import MCTSPlayer
from player2 import AdvancedHexPlayer
import time
import os

def clear_console():
    """Limpia la consola para actualizar el tablero en una sola posición."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_board(board: HexBoard):
    """Imprime el tablero en consola con forma de rombo, colores y numeración."""
    size = len(board.board)
    clear_console()
    print("   ", end="")
    for col in range(size):
        print(f"\033[93m{col}\033[0m ", end="")  # Numeración de columnas en amarillo
    print()
    for i, row in enumerate(board.board):
        print(f"\033[93m{i}\033[0m ", end="")  # Numeración de filas en amarillo
        print(" " * (size - i - 1), end="")  # Espacios iniciales para el desplazamiento
        for cell in row:
            if cell == 1:
                print("\033[91m🔴\033[0m", end=" ")  # Rojo para el jugador 1
            elif cell == 2:
                print("\033[94m🔵\033[0m", end=" ")  # Azul para el jugador 2
            else:
                print("⬜", end=" ")  # Blanco para celdas vacías
        print()  # Nueva línea al final de cada fila
    print()

def simulate_game(board_size: int, move_timeout: float, human_vs_ai: bool):
    """Simula una partida de Hex entre un humano y una IA o entre dos IA."""
    board = HexBoard(board_size)
    player1 = AdvancedHexPlayer(player_id=1)
    player2 = AdvancedHexPlayer(player_id=2)
    players = [player1, player2]
    current_player_index = 0
    history = []  # Historial de jugadas

    print("Comenzando la simulación de la partida...\n")
    print_board(board)

    while True:
        current_player = players[current_player_index]

        if human_vs_ai and current_player.player_id == 2:
            # Turno del humano
            while True:
                try:
                    row = int(input("Ingrese la fila donde desea jugar: "))
                    col = int(input("Ingrese la columna donde desea jugar: "))
                    if 0 <= row < board.size and 0 <= col < board.size and board.board[row][col] == 0:
                        move = (row, col)
                        break
                    else:
                        print("Movimiento inválido. Intente nuevamente.")
                except ValueError:
                    print("Entrada inválida. Intente nuevamente.")

        else:
            # Turno de la IA
            move = current_player.play(board,move_timeout)
            if not move:
                print(f"Jugador {current_player.player_id} no puede realizar más movimientos.")
                break

        board.place_piece(move[0], move[1], current_player.player_id)
        history.append((current_player.player_id, move))  # Registrar jugada en el historial
        print(f"Jugador {current_player.player_id} juega en {move}.")
        print_board(board)

        # Verificar si el jugador actual ha ganado
        if board.check_connection(current_player.player_id):
            print(f"¡Jugador {current_player.player_id} ha ganado la partida!")
            break

        # Cambiar al siguiente jugador
        current_player_index = 1 - current_player_index

        # Pausa para observar el progreso
        time.sleep(0.5)

    print("\nHistorial de jugadas:")
    for player_id, move in history:
        print(f"Jugador {player_id} jugó en {move}.")

if __name__ == "__main__":
    while True:
        try:
            board_size = int(input("Ingrese el tamaño del tablero (ejemplo: 11): "))
            move_timeout = float(input("Ingrese el tiempo límite por jugada en segundos (ejemplo: 5.0): "))
            mode = input("¿Desea jugar contra la IA? (s/n): ").strip().lower()
            human_vs_ai = mode == 's'
            if board_size > 0 and move_timeout > 0:
                break
            else:
                print("Por favor, ingrese valores válidos.")
        except ValueError:
            print("Entrada inválida. Intente nuevamente.")

    simulate_game(board_size, move_timeout, human_vs_ai)


# new_board = self.board.clone()
#         new_board.place_piece(*move, self.board.current_player)
#         child = MCTSNode(new_board, self, move)
#         self.children.append(child)