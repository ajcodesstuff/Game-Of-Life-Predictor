import numpy as np
from collections import deque
import pickle

# Constants
WIDTH, HEIGHT = 400, 400
CELL_SIZE = 20
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
FPS = 60  # Increased for faster simulation

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRID_LINE = (50, 50, 50)

class GameOfLife:
    def __init__(self):
        self.datacount = 0
        p = np.random.uniform(0.01, 0.99)
        self.grid = np.random.choice([0, 1], size=(GRID_HEIGHT, GRID_WIDTH), p=[1-p, p])
        self.initial_grid = self.grid.copy()
        self.prev_states = deque(maxlen=100)
        self.running = True
        self.dataset = []

    def count_neighbors(self, x, y):
        neighbors = [
            (x-1, y-1), (x-1, y), (x-1, y+1),
            (x, y-1),             (x, y+1),
            (x+1, y-1), (x+1, y), (x+1, y+1)
        ]
        count = 0
        for nx, ny in neighbors:
            if 0 <= nx < GRID_HEIGHT and 0 <= ny < GRID_WIDTH:
                count += self.grid[nx, ny]
        return count

    def update_grid(self):
        current_state = self.grid.tobytes()

        # --- STEP 1: compute next grid ---
        new_grid = np.zeros_like(self.grid)

        for x in range(GRID_HEIGHT):
            for y in range(GRID_WIDTH):
                neighbors = self.count_neighbors(x, y)

                if self.grid[x, y] == 1:
                    if neighbors in [2, 3]:
                        new_grid[x, y] = 1
                else:
                    if neighbors == 3:
                        new_grid[x, y] = 1

        new_state = new_grid.tobytes()

        # --- STEP 2: initialize tracking if not present ---
        if not hasattr(self, "seen_states"):
            self.seen_states = set()
            self.steps = 0

        label = None

        # Death condition
        if np.count_nonzero(new_grid) == 0:
            label = "Dead"

        # Stable (no change)
        elif new_state == current_state and self.steps >= 2:
            label = "Stable"

        # Oscillation (repeat but not identical step)
        elif new_state in self.seen_states and self.steps >= 2:
            label = "Oscillating"

        # --- STEP 4: if labeled, save and reset ---
        if label:
            self.dataset.append((self.initial_grid.copy(), label))

            # reset simulation
            p = np.random.uniform(0.1, 0.9)
            self.grid = np.random.choice([0, 1], size=(GRID_HEIGHT, GRID_WIDTH), p=[1-p, p])
            self.initial_grid = self.grid.copy()

            self.seen_states.clear()
            self.steps = 0

            return  # important: stop here after reset

        # --- STEP 5: continue simulation ---
        self.seen_states.add(current_state)
        self.grid = new_grid
        self.steps += 1

    def run(self):
        while True:
            self.update_grid()
            CHUNK_SIZE = 100
            TARGET_SAMPLES = 9000

            if len(self.dataset) >= CHUNK_SIZE:
                with open('test_dataset.pkl', 'ab') as f:
                    pickle.dump(self.dataset, f)

                print(f"Saved chunk of {len(self.dataset)} samples")

                self.datacount += len(self.dataset)
                self.dataset.clear()

                if self.datacount >= TARGET_SAMPLES:
                    print("Data collection complete.")
                    break
                    
if __name__ == "__main__":
    game = GameOfLife()
    game.run()