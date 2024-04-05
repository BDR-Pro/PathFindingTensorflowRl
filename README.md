# Pathfinding with A* and Q-Learning

This project showcases a pathfinding visualization using A* algorithm and Q-Learning model to find paths through a generated maze. The comparison between these two methods provides insights into their efficiency and applicability in solving pathfinding problems. Utilizing Pygame for visualization and TensorFlow for the Q-Learning model, this project offers an engaging way to understand and analyze pathfinding algorithms and machine learning in action.

![alt text](Path_Finding.keras(1).png)

## Features

- **Maze Generation**: Randomly generates a maze for pathfinding.
- **A* Algorithm Implementation**: Utilizes the A* algorithm to find the shortest path from start to goal.
- **Q-Learning Model**: Employs a Q-Learning model developed with TensorFlow to learn and find paths through the maze.
- **Visualization**: Uses Pygame for real-time visualization of the maze, paths found by A* and the Q-Learning model, start and goal positions.
- **GPU Acceleration**: Checks for CUDA compatibility to leverage GPU acceleration for the Q-Learning model training and inference, enhancing performance.
- **Threaded Execution**: Runs pathfinding operations in separate threads to keep the UI responsive and provide real-time updates.

## Installation

Before you start, ensure you have Python 3.7+ and pip installed on your system. Clone this repository or download the code.

To install the required packages, run:

```bash
pip install pygame tensorflow keras numpy
```

## Usage

To start the application, navigate to the project directory and run:

```bash
python pathfinding.py
```

The Pygame window will open, displaying the generated maze and, eventually, the paths found by both the A* algorithm and the Q-Learning model.

![alt text](<لقطة شاشة 2024-04-05 034314.png>)

## Development

This project uses:

- Python 3.7+
- TensorFlow 2.x for the Q-Learning model.
- Pygame for visualization.
- NumPy for array manipulation and operations.

## Contributions

Contributions are welcome! Whether it's bug reports, improvements, or new feature suggestions, feel free to open an issue or a pull request.

---

Dive into the fascinating world of pathfinding algorithms and reinforcement learning with this interactive visualization project!
