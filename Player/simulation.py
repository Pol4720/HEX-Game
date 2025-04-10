from player import AIPlayer, HexBoard
import time
import os

def print_board(board: HexBoard):
    """Imprime el tablero en consola con colores para las jugadas."""
    for row in board.board:
        for cell in row:
            if cell == 1:
                print("\033[91m🔴\033[0m", end=" ")  # Rojo para el jugador 1
            elif cell == 2:
                print("\033[94m🔵\033[0m", end=" ")  # Azul para el jugador 2
            else:
                print("⬜", end=" ")  # Blanco para celdas vacías
        print()
    print()

def simulate_game(board_size: int, move_timeout: float):
    """Simula una partida de Hex entre dos instancias de AIPlayer."""
    board = HexBoard(board_size)
    player1 = AIPlayer(player_id=1, timeout=move_timeout)
    player2 = AIPlayer(player_id=2, timeout=move_timeout)
    players = [player1, player2]
    current_player_index = 0

    print("Comenzando la simulación de la partida...\n")
    print_board(board)

    while True:
        current_player = players[current_player_index]
        move = current_player.play(board)
        if not move:
            print(f"Jugador {current_player.player_id} no puede realizar más movimientos.")
            break

        board.place_piece(move[0], move[1], current_player.player_id)
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

if __name__ == "__main__":
    while True:
        try:
            board_size = int(input("Ingrese el tamaño del tablero (ejemplo: 11): "))
            move_timeout = float(input("Ingrese el tiempo límite por jugada en segundos (ejemplo: 5.0): "))
            if board_size > 0 and move_timeout > 0:
                break
            else:
                print("Por favor, ingrese valores válidos.")
        except ValueError:
            print("Entrada inválida. Intente nuevamente.")

    simulate_game(board_size, move_timeout)
