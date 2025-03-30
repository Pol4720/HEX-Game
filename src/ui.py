import pygame
from game import HexGame

class HexGameUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.game = HexGame(size=11)

    def show_menu(self):
            font = pygame.font.Font(None, 36)
            menu_running = True
            game_mode = None
            board_size = 11
            difficulty = "Easy"

            while menu_running:
                self.screen.fill((200, 200, 200))
                title = font.render("HEX Game Menu", True, (0, 0, 0))
                self.screen.blit(title, (300, 50))

                # Game mode selection
                pvp_text = font.render("1. Player vs Player", True, (0, 0, 0))
                self.screen.blit(pvp_text, (100, 150))
                pve_text = font.render("2. Player vs AI", True, (0, 0, 0))
                self.screen.blit(pve_text, (100, 200))

                # Board size selection
                size_text = font.render(f"Board Size: {board_size}x{board_size} (Press UP/DOWN)", True, (0, 0, 0))
                self.screen.blit(size_text, (100, 300))

                # Difficulty selection
                difficulty_text = font.render(f"Difficulty: {difficulty} (Press LEFT/RIGHT)", True, (0, 0, 0))
                self.screen.blit(difficulty_text, (100, 350))

                # Play button
                play_text = font.render("Press ENTER to Play", True, (0, 0, 0))
                self.screen.blit(play_text, (300, 450))

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        menu_running = False
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_1:
                            game_mode = "PvP"
                        elif event.key == pygame.K_2:
                            game_mode = "PvE"
                        elif event.key == pygame.K_UP:
                            board_size = min(board_size + 2, 15)
                        elif event.key == pygame.K_DOWN:
                            board_size = max(board_size - 2, 7)
                        elif event.key == pygame.K_LEFT:
                            difficulty = {"Easy": "Intermediate", "Intermediate": "Hard", "Hard": "Extreme", "Extreme": "Easy"}[difficulty]
                        elif event.key == pygame.K_RIGHT:
                            difficulty = {"Easy": "Extreme", "Extreme": "Hard", "Hard": "Intermediate", "Intermediate": "Easy"}[difficulty]
                        elif event.key == pygame.K_RETURN:
                            menu_running = False

                pygame.display.flip()
                self.clock.tick(30)

            self.game = HexGame(size=board_size, mode=game_mode, difficulty=difficulty)
        

    def run(self):
        self.show_menu()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.screen.fill((255, 255, 255))
            # ...existing code for rendering...
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()