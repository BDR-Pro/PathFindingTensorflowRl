import heapq  # Priority queue for A*
import pygame  # For GUI visualization
import warnings
from tensorflow import keras
from tensorflow import config
from tensorflow import test as test_tensor
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
import threading
import os 
from functools import partial
import cProfile
import sys
import numpy as np

global_a_star_path = []
global_tf_model_path = []
path_to_draw_lock = threading.Lock()
warnings.filterwarnings("ignore")
# Check if TensorFlow can access the GPU
gpus = config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            config.experimental.set_memory_growth(gpu, True)
        logical_gpus = config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("Not running on CUDA. GPUs not found.")


class Node:
    def __init__(self, position, parent=None, g_score=0, h_score=0):
        self.position = position
        self.parent = parent
        self.g_score = g_score  # Cost from start to current node
        self.h_score = h_score  # Heuristic estimate of cost to goal
        self.f_score = self.g_score + self.h_score

    def __lt__(self, other):
        return self.f_score < other.f_score

def a_star_search(maze, start, goal):
    open_set = []
    closed_set = set()
    came_from = {}

    start_node = Node(start)
    start_node.h_score = manhattan_distance(start, goal)  # Heuristic example
    heapq.heappush(open_set, start_node)

    while open_set:
        current = heapq.heappop(open_set)
        closed_set.add(current.position)

        if current.position == goal:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Reverse for start to goal order

        for neighbor in get_neighbors(maze, current.position):
            if neighbor in closed_set:
                continue

            tentative_g_score = current.g_score + 1  # Assuming uniform movement cost

            if neighbor not in (node.position for node in open_set):
                new_node = Node(neighbor, current, tentative_g_score)
                new_node.h_score = manhattan_distance(neighbor, goal)
                heapq.heappush(open_set, new_node)
                came_from[neighbor] = current
            elif tentative_g_score < get_node(open_set, neighbor).g_score:
                came_from[neighbor] = current
                new_node = get_node(open_set, neighbor)
                new_node.parent = current
                new_node.g_score = tentative_g_score
                heapq.heapify(open_set)  # Update priority in heap

    return None  # No path found

def get_neighbors(maze, position):
    # Modify this based on your maze representation (e.g., up, down, left, right)
    neighbors = []
    rows, cols = len(maze), len(maze[0])
    dx = [-1, 0, 1, 0]
    dy = [0, -1, 0, 1]
    for i in range(4):
        x = position[0] + dx[i]
        y = position[1] + dy[i]
        if 0 <= x < rows and 0 <= y < cols and maze[x][y] != 1:
            neighbors.append((x, y))
    return neighbors

def get_node(open_set, position):
    for node in open_set:
        if node.position == position:
            return node
    return None

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def is_loadable(filename):
    #search is parent dir for file
    #search in current dir
    if os.path.isfile(os.path.join(os.getcwd(), filename)):
        return True
    return False

class QLearningAgent:
    def __init__(self, maze, alpha=0.1, gamma=0.9, state_size=2, action_size=4):
        self.maze = maze
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.state_size = state_size  # The size of the input to the model (e.g., position in maze)
        self.action_size = action_size  # The number of actions the agent can take
        
        # Load or initialize model
        self.model = self.load_or_initialize_model()

    def load_or_initialize_model(self):
        if is_loadable("Path_Finding.keras"):
            model = keras.models.load_model("Path_Finding.keras")
            print("Model loaded successfully.")
        else:
            is_cuda = test_tensor
            if is_cuda:
                print(f"CUDA is available.\n {is_cuda=}")
            else:
                print("CUDA is not available.")
            print("Initializing new model.")
            inputs = Input(shape=(self.state_size,))
            x = Dense(64, kernel_initializer='he_uniform')(inputs)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dense(64, kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dense(64, kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dense(64, kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dense(64, kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dense(64, kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)  # Add dropout for regularization
            outputs = Dense(self.action_size, activation='linear')(x)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss=keras.losses.MeanSquaredError(), optimizer=Adam(learning_rate=self.alpha))

            return model
    def get_action(self, state):
        """ Get the action with the highest Q-value for the current state."""
        state = np.array(state).reshape(-1, self.state_size)
        q_values = self.model.predict(state)[0]
        return np.argmax(q_values)  # Choose action with highest Q-value for the current state
    
    def train(self, experiences, epochs=1):
        """ Train the model using the given experiences.

        Args:
            experiences (list): A list of experiences, each containing a state, action, reward, next state, and done flag.
            epochs (int): The number of epochs to train the model.
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.array(states)
        next_states = np.array(next_states)

        # Predict Q-values for starting and next states
        q_values = self.model.predict(states)
        q_values_next = self.model.predict(next_states)
        
        # Update Q-values
        for i in range(len(q_values)):
            if dones[i]:
                q_values[i, actions[i]] = rewards[i]
            else:
                q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(q_values_next[i])
        
        # Train the model
        self.model.fit(states, q_values, epochs=epochs, verbose=0)
        self.model.save("Path_Finding.keras")
        
    def test(self, start, goal):
        current_state = start
        path = [start]  # Initialize path with the starting position
        steps = 0
        
        # Loop until the goal is reached or the maximum number of steps is exceeded
        while current_state != goal :
            action = self.get_action(current_state)  # Predict the best action for the current state
            
            # Convert action into a change in position
            # Assuming actions are [0: up, 1: right, 2: down, 3: left]
            if action == 0:  # Up
                next_state = (current_state[0] - 1, current_state[1])
            elif action == 1:  # Right
                next_state = (current_state[0], current_state[1] + 1)
            elif action == 2:  # Down
                next_state = (current_state[0] + 1, current_state[1])
            elif action == 3:  # Left
                next_state = (current_state[0], current_state[1] - 1)
            
            # Check if the next state is within the maze bounds and not a wall
            if 0 <= next_state[0] < len(self.maze) and 0 <= next_state[1] < len(self.maze[0]) and self.maze[next_state[0]][next_state[1]] == 0:
                current_state = next_state  # Update the current state
                path.append(next_state)  # Append the new state to the path
            else:
                print("Encountered a wall or out-of-bounds try again -- after puinshing the model.")
                # Penalize the model for hitting a wall or going out of bounds
                reward = -1
                done = True
                self.train([(current_state, action, reward, current_state, done)], epochs=10)
                
                
            
            steps += 1
        
        if current_state == goal:
            print("Goal reached!")
            
        return path
    



    

def generate_maze(rows=100, cols=100, wall_prob=.4):
    # Simplified maze generation for illustration. Adjust as per your actual implementation.
    """Generates a random maze with the given dimensions and wall probability."""
    return np.random.choice([0, 1], size=(rows, cols), p=[1-wall_prob, wall_prob])



def draw_maze(screen, maze, cell_size,start,goal):
    """ Draw the maze on the Pygame screen."""
    rows, cols = len(maze), len(maze[0])
    for i in range(rows):
        for j in range(cols):
            color = (255, 255, 255) if maze[i][j] == 0 else (0, 0, 0)
            pygame.draw.rect(screen, color, (j * cell_size, i * cell_size, cell_size, cell_size))
    pygame.draw.rect(screen, (255, 255, 0), (goal[1] * cell_size, goal[0] * cell_size, cell_size, cell_size))
    pygame.draw.rect(screen, (128, 0, 128), (start[1] * cell_size, start[0] * cell_size, cell_size, cell_size))
            
def draw_path(screen, path, cell_size, color=(255, 0, 0)):
    # Visualize the path
    """ Draw the path on the Pygame screen."""
    for position in path:
        pygame.draw.rect(screen, color, (position[1] * cell_size, position[0] * cell_size, cell_size, cell_size))


def run_training_and_testing(agent, start, goal, maze, update_gui_callback=None):
    """
    This function runs the training and testing in a separate thread.
    Once training and testing are done, it can optionally call a callback function
    to update the GUI with the new path.
    """
    experiences = collect_experiences(agent, start, goal, maze)
    agent.train(experiences)
    path = agent.test(start, goal)
    if update_gui_callback:
        update_gui_callback(path)


def update_gui_with_path(path, path_type, lock):
    """ Update the global path variable based on the path type."""
    global global_a_star_path, global_tf_model_path
    with lock:
        if path_type == 'a_star':
            global_a_star_path = path
        elif path_type == 'tf_model':
            global_tf_model_path = path



def random_goal(maze):
    # Using a previously discussed function to pick a non-wall cell randomly as the goal
    """ Pick a random non-wall cell in the maze as the goal. """
    rows, cols = maze.shape
    free_cells = np.argwhere(maze == 0)
    return tuple(free_cells[np.random.randint(len(free_cells))])

def take_action(maze, current_state, action):
    """
    Calculate the next state based on the current state and action.
    
    Parameters:
    - maze: 2D list representing the maze, where 1's are walls and 0's are paths.
    - current_state: Tuple of (row, col) indicating the agent's current position.
    - action: Integer representing the direction the agent moves.
              Assuming 0: up, 1: right, 2: down, 3: left.
    
    Returns:
    - Tuple of (row, col) for the next state.
    """
    # Extract the current position
    row, col = current_state
    
    # Determine the next position based on the action
    if action == 0:  # Up
        next_state = (max(row - 1, 0), col)
    elif action == 1:  # Right
        next_state = (row, min(col + 1, len(maze[0]) - 1))
    elif action == 2:  # Down
        next_state = (min(row + 1, len(maze) - 1), col)
    elif action == 3:  # Left
        next_state = (row, max(col - 1, 0))
    
    # Check if the next state is a wall or outside the maze boundaries
    # If it's a wall, the agent stays in the current state
    if maze[next_state[0]][next_state[1]] == 1:
        return current_state  # Stay in place if the next state is a wall
    else:
        return next_state  # Move to the next state if it's not a wall

def collect_experiences(agent, start, goal, maze, num_experiences=100):
    """Collect experiences for training the agent."""
    experiences = []
    while len(experiences) < num_experiences:
        current_state = start
        steps = 0
        while current_state != goal and steps < 100 and len(experiences) < num_experiences:
            action = agent.get_action(current_state)
            next_state = take_action(maze, current_state, action)
            reward = 1 if next_state == goal else 0
            done = next_state == goal
            experiences.append((current_state, action, reward, next_state, done))
            current_state = next_state
            steps += 1

    return experiences

def find_path_a_star(maze, start, goal, update_callback):
    path = a_star_search(maze, start, goal)
    update_callback(path)  # 'path_type' and 'lock' are already set by partial

def find_path_tf_model(agent, start, goal, update_callback):
    path = agent.test(start, goal)
    update_callback(path)  # 'path_type' and 'lock' are already set by partial


def thread_callback(update_func, path, path_type):
    update_func(path, path_type, path_to_draw_lock)
    
def main():
    # Initialization code...

    # Start Pygame
    pygame.init()
    cols, rows = 100, 100
    cell_size = 6
    screen = pygame.display.set_mode((cols * cell_size, rows * cell_size), pygame.RESIZABLE)
    pygame.display.set_caption("Pathfinding Visualization with A* and TensorFlow Model (Q-Learning) Paths")
    clock = pygame.time.Clock()
    # Initialize paths and agent positions
    maze = generate_maze(rows, cols)
    start, goal = (0, 0), random_goal(maze)
    agent = QLearningAgent(maze)

    # Correct usage of partial for your scenario
    update_a_star = partial(update_gui_with_path, path_type='a_star', lock=path_to_draw_lock)
    update_tf_model = partial(update_gui_with_path, path_type='tf_model', lock=path_to_draw_lock)

    # Then, you pass these partially applied functions to your threads WITHOUT directly passing the arguments in args
    a_star_thread = threading.Thread(target=find_path_a_star, args=(maze, start, goal, update_a_star))
    tf_model_thread = threading.Thread(target=find_path_tf_model, args=(agent, start, goal, update_tf_model))

    a_star_thread.start()
    tf_model_thread.start()

    # Main Pygame loop
    running = True
    a_star_complete, tf_model_complete = False, False
    while running and not (a_star_complete and tf_model_complete):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear screen
        screen.fill((0, 0, 0))

        # Draw the maze
        draw_maze(screen, maze, cell_size, start, goal)

        # Inside your main loop:
        with path_to_draw_lock:
            if global_a_star_path:
                draw_path(screen, global_a_star_path, cell_size, color=(255, 0, 0))  # Red for A*
            if global_tf_model_path:
                draw_path(screen, global_tf_model_path, cell_size, color=(0, 255, 0))  # Green for TF model


        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Clean up
    pygame.quit()
    a_star_thread.join()
    tf_model_thread.join()


if __name__ == "__main__":
    profile_run = False
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        profile_run = True
    
    if profile_run:
        cProfile.run('main()', sort='time')
    else:
        main()