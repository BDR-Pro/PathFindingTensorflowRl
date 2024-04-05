import heapq  
import pygame  
import warnings
#from tensorflow import keras
#from tensorflow import config
#from keras.models import Model
#from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
#from keras.optimizers import Adam
import os 
import numpy as np
import sys
from functools import wraps
import time

def time_in_milliseconds_or_seconds(time):
    return f"{time:.2f} seconds" if time > 1 else f"{time * 1000:.2f} milliseconds"

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_time = time_in_milliseconds_or_seconds(total_time)
        print(f'Function {func.__name__} Took {total_time} .')
        return result
    return timeit_wrapper

warnings.filterwarnings("ignore")

"""
gpus = config.list_physical_devices('GPU')
if gpus:
    try:
        
        for gpu in gpus:
            config.experimental.set_memory_growth(gpu, True)
        logical_gpus = config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        
        print(e)
else:
    print("Not running on CUDA. GPUs not found.")

"""

class Node:
    """
    Represents a node in the search space.
    """

    def __init__(self, position, parent, g_score, h_score):
        self.position = position
        self.parent = parent
        self.g_score = g_score  # Cost from start to this node
        self.h_score = h_score  # Heuristic estimate of cost from this node to goal
        self.f_score = g_score + h_score  # Total estimated cost (f = g + h)

    def __lt__(self, other):
        """
        Overload comparison for priority queue (prioritize lower f_score).
        """
        return self.f_score < other.f_score

    def __eq__(self, other):
        return self.position == other.position


def removelastline():
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
 
@timeit
def a_star_search(maze, start, goal, limit=1_000_000):
    """
    Implements the A* search algorithm to find the shortest path through a maze.

    Args:
        maze: A 2D list representing the maze, where 0 indicates a walkable cell and 1 indicates an obstacle.
        start: A tuple representing the starting position (row, col).
        goal: A tuple representing the goal position (row, col).
        limit: An integer limiting the number of iterations (optional, defaults to 1 million).

    Returns:
        A list of tuples representing the shortest path from start to goal, or None if no path is found.
    """

    open_set = []
    closed_set = set()

    # Create the starting node with accurate f_score
    start_node = Node(start, None, 0, manhattan_distance(start, goal))
    heapq.heappush(open_set, (start_node.f_score, start_node))  # Push (f_score, node)

    counter = 0

    while open_set and counter < limit:
        counter += 1

        # Prioritize node with lowest f_score
        current_f_score, current_node = heapq.heappop(open_set)
        removelastline()
        print(f"Current node: {current_node.position} -- {current_node.f_score} -- {counter} iterations.")
        
        if current_node.position == goal:
            path = []
            print("Path found! A* Algorthim Building path.")
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Reverse for final path

        closed_set.add(current_node.position)

        for neighbor_position in get_neighbors(maze, current_node.position):
            if neighbor_position in closed_set:
                continue

            g_score = current_node.g_score + 1
            h_score = manhattan_distance(neighbor_position, goal)
            f_score = g_score + h_score

            # **Fixed line: Define neighbor node with correct arguments**
            neighbor_node = Node(neighbor_position, current_node, g_score, h_score)

            if neighbor_position not in (node for node in open_set):
                # New neighbor: add to open set with accurate f_score
                heapq.heappush(open_set, (f_score, neighbor_node))
            else:
                # Existing neighbor: update if new path has lower f_score
                for existing_node in open_set:
                    if existing_node.position == neighbor_position and f_score < existing_node.f_score:
                        existing_node.parent = current_node
                        existing_node.g_score = g_score
                        existing_node.f_score = f_score
                        break  # Only update once per neighbor

    return None  # No path found

def get_neighbors(maze, position):
    
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
    
    
    if os.path.isfile(os.path.join(os.getcwd(), filename)):
        return True
    return False

class QLearningAgent:
    def __init__(self, maze, alpha=0.1, gamma=0.9, state_size=2, action_size=4):
        self.maze = maze
        self.alpha = alpha  
        self.gamma = gamma  
        self.state_size = state_size  
        self.action_size = action_size  
        self.train_calls = 0 
        """
        self.model = self.load_or_initialize_model()

    def load_or_initialize_model(self):
        if is_loadable("Path_Finding.keras"):
            model = keras.models.load_model("Path_Finding.keras")
            print("Model loaded successfully.")
        else:
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
            x = Dropout(0.5)(x)  
            outputs = Dense(self.action_size, activation='linear')(x)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss=keras.losses.MeanSquaredError(), optimizer=Adam(learning_rate=self.alpha))

            return model
            """
    def get_action(self, state):
        """ Get the action with the highest Q-value for the current state."""
        state = np.array(state).reshape(-1, self.state_size)
        q_values = self.model.predict(state)[0]
        return np.argmax(q_values)  
    
    def train(self, experiences, epochs=1):
        """ Train the model using the given experiences.

        Args:
            experiences (list): A list of experiences, each containing a state, action, reward, next state, and done flag.
            epochs (int): The number of epochs to train the model.
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.array(states)
        next_states = np.array(next_states)

        
        q_values = self.model.predict(states)
        q_values_next = self.model.predict(next_states)
        self.train_calls += 1
        print(f"Training call {self.train_calls}")
        
        for i in range(len(q_values)):
            if dones[i]:
                q_values[i, actions[i]] = rewards[i]
            else:
                q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(q_values_next[i])
        
        
        self.model.fit(states, q_values, epochs=epochs, verbose=0)
        
        self.model.save("Path_Finding.keras")
        
    def test(self, start, goal, limit=1000):
        current_state = start
        path = [start]  
        steps = 0
        counter=0
        
        while current_state != goal and counter < limit:
            counter += 1
            action = self.get_action(current_state)  
            
            
            
            if action == 0:  
                next_state = (current_state[0] - 1, current_state[1])
            elif action == 1:  
                next_state = (current_state[0], current_state[1] + 1)
            elif action == 2:  
                next_state = (current_state[0] + 1, current_state[1])
            elif action == 3:  
                next_state = (current_state[0], current_state[1] - 1)
            
            
            if 0 <= next_state[0] < len(self.maze) and 0 <= next_state[1] < len(self.maze[0]) and self.maze[next_state[0]][next_state[1]] == 0:
                current_state = next_state  
                path.append(next_state)  
            else:
                print("Encountered a wall or out-of-bounds try again -- after puinshing the model.")
                
                reward = -1
                done = True
                self.train([(current_state, action, reward, current_state, done)], epochs=10)
                
                
            
            steps += 1
        
        if current_state == goal:
            print("Goal reached!")
            
        return path
    



    

def generate_maze(rows=100, cols=100, wall_prob=.25):
    print(f"Generating maze with diffuclty == {wall_prob}.")
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
    
    """ Draw the path on the Pygame screen."""
    try:
        #skip the first element of the path, as it is the start position and the last element as it is the goal position
        for position in path [1:-1]:
            # Draw with a little delay to visualize the path
            pygame.draw.rect(screen, color, (position[1] * cell_size, position[0] * cell_size, cell_size, cell_size))
    except TypeError:
        print("No path found in draw path.")
        print("Path:",path)
        print("Maze is unsolvable.")
    return False

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




def random_goal(maze):
    
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
    
    row, col = current_state
    
    
    if action == 0:  
        next_state = (max(row - 1, 0), col)
    elif action == 1:  
        next_state = (row, min(col + 1, len(maze[0]) - 1))
    elif action == 2:  
        next_state = (min(row + 1, len(maze) - 1), col)
    elif action == 3:  
        next_state = (row, max(col - 1, 0))
    
    
    
    if maze[next_state[0]][next_state[1]] == 1:
        return current_state  
    else:
        return next_state  

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


    
def main():
    
    pygame.init()
    
    maze = generate_maze()
    print("Maze generated successfully.")
    start = (0, 0)
    goal = random_goal(maze)
    print("Goal generated successfully.\n",goal)
    cell_size = 6
    screen = pygame.display.set_mode((len(maze[0]) * cell_size, len(maze) * cell_size))
    clock = pygame.time.Clock()
    agent = QLearningAgent(maze)
    

    
    
    a_star_path = a_star_search(maze, start, goal)  
    #tf_model_path = agent.test(start, goal)  
    if a_star_path:
        print(f"A* Path successfully found with {len(a_star_path)-2} steps.")   

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        
        
        screen.fill((0, 0, 0))
        draw_maze(screen, maze, cell_size,start,goal)
        draw_path(screen, a_star_path, cell_size, (255, 0, 0))

        #draw_path(screen, tf_model_path, cell_size, (0, 255, 0))

        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


while True:
    main()