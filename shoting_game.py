import pygame
import random
import pickle

# ────────────────────────────
# Game constants
# ────────────────────────────
WIDTH, HEIGHT   = 600, 600
GRID_SIZE       = 10
CELL_SIZE       = WIDTH // GRID_SIZE

# RL settings
ACTIONS         = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SHOOT']
LEARNING_RATE   = 0.1
DISCOUNT        = 0.95
EPSILON         = 0.1

WIN_SCORE       = 10          # first to 10 wins

# ────────────────────────────
# Pygame setup
# ────────────────────────────
pygame.init()
pygame.font.init()
FONT_SMALL = pygame.font.SysFont("arial", 24)
FONT_LARGE = pygame.font.SysFont("arial", 48)

background_img = pygame.image.load("space.png")
background_img = pygame.transform.scale(background_img, (WIDTH, HEIGHT))

win   = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Tank Battle - Player vs AI")

# ────────────────────────────
# Tank class
# ────────────────────────────
class Tank:
    COOLDOWN_MS = 1000           # 1 s between shots

    def __init__(self, x, y, is_player=False):
        self.x = x
        self.y = y
        self.is_player = is_player
        self.color   = (0, 255, 0) if is_player else (255, 0, 0)
        self.bullets = []
        self.last_shot_time = 0

    def move(self, direction):
        if direction == 'UP'    and self.y > 0:               self.y -= 1
        if direction == 'DOWN'  and self.y < GRID_SIZE-1:     self.y += 1
        if direction == 'LEFT'  and self.x > 0:               self.x -= 1
        if direction == 'RIGHT' and self.x < GRID_SIZE-1:     self.x += 1

    def can_shoot(self):
        return pygame.time.get_ticks() - self.last_shot_time >= self.COOLDOWN_MS

    def shoot(self):
        if self.can_shoot():
            self.bullets.append([
                self.x,
                self.y - 1 if self.is_player else self.y + 1
            ])
            self.last_shot_time = pygame.time.get_ticks()

    def update_bullets(self):
        for b in self.bullets:
            b[1] += -1 if self.is_player else 1
        self.bullets = [b for b in self.bullets if 0 <= b[1] < GRID_SIZE]

    def draw(self):
        pygame.draw.rect(
            win, self.color,
            (self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )
        for bx, by in self.bullets:
            pygame.draw.circle(
                win, self.color,
                (bx * CELL_SIZE + CELL_SIZE//2, by * CELL_SIZE + CELL_SIZE//2),
                5
            )

# ────────────────────────────
# Q-learning agent
# ────────────────────────────
class QLearningAgent:
    def __init__(self):
        self.q_table = {}

    def get_state(self, ai, player):
        return (ai.x, ai.y, player.x, player.y)

    def choose_action(self, state):
        if random.random() < EPSILON or state not in self.q_table:
            return random.choice(ACTIONS)
        return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, old_s, action, reward, new_s):
        for s in (old_s, new_s):
            if s not in self.q_table:
                self.q_table[s] = {a: 0 for a in ACTIONS}

        old_q     = self.q_table[old_s][action]
        future_q  = max(self.q_table[new_s].values())
        updated_q = old_q + LEARNING_RATE * (reward + DISCOUNT*future_q - old_q)
        self.q_table[old_s][action] = updated_q

agent = QLearningAgent()

# ────────────────────────────
# Main game loop
# ────────────────────────────
def main():
    player      = Tank(5, 9, True)
    ai          = Tank(5, 0)
    player_pts  = 0
    ai_pts      = 0
    run         = True

    while run:
        clock.tick(10)
        win.blit(background_img, (0, 0))

        # ── Events ───────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # ── Player input ─────────────────
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] : player.move('LEFT')
        if keys[pygame.K_RIGHT]: player.move('RIGHT')
        if keys[pygame.K_UP]   : player.move('UP')
        if keys[pygame.K_DOWN] : player.move('DOWN')
        if keys[pygame.K_SPACE]: player.shoot()

        # ── AI decision ──────────────────
        state  = agent.get_state(ai, player)
        action = agent.choose_action(state)
        if action in {'UP','DOWN','LEFT','RIGHT'}: ai.move(action)
        if action == 'SHOOT':                      ai.shoot()

        # ── Update bullets ───────────────
        player.update_bullets()
        ai.update_bullets()

        # ── Hit detection & scoring ──────
        for b in ai.bullets[:]:
            if b[0] == player.x and b[1] == player.y:
                ai.bullets.remove(b)
                ai_pts += 1
                print("AI hit Player")
                break

        for b in player.bullets[:]:
            if b[0] == ai.x and b[1] == ai.y:
                player.bullets.remove(b)
                player_pts += 1
                print("Player hit AI")
                break

        # ── RL learning ──────────────────
        agent.learn(state, action, 0, agent.get_state(ai, player))

        # ── Draw entities ────────────────
        player.draw()
        ai.draw()

        score_surf = FONT_SMALL.render(
            f"Player: {player_pts}   AI: {ai_pts}", True, (255,255,255)
        )
        win.blit(score_surf, (10, 10))

        # ── Win check ────────────────────
        if player_pts >= WIN_SCORE or ai_pts >= WIN_SCORE:
            winner = "Player Wins!" if player_pts >= WIN_SCORE else "AI Wins!"
            text   = FONT_LARGE.render(winner, True, (255,255,0))
            win.blit(
                text,
                (WIDTH//2 - text.get_width()//2,
                 HEIGHT//2 - text.get_height()//2)
            )
            pygame.display.update()
            pygame.time.delay(3000)
            run = False
            continue

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
