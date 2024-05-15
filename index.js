const express=  require("express")
const app = express()
const bodyParser = require('body-parser');
const port = 8000
const cors= require('cors')
app.use(cors())

app.use(bodyParser.json({ limit: '50mb' }));

app.use(express.json())
const API_KEY= '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'
app.post('/questions/:topic/:level', async (req, res) => {
  const { topic, level } = req.params;
app.get("/",(req,res)=>{
  res.send("")
})

  const options = {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'gpt-3.5-turbo',
      messages: [{
        role: 'user',
        content: `
          ///////nqueens////////////////////////////////////////////////
def is_safe(board, row, col, N):
    # Check the column on the current row
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper diagonal on right side
    for i, j in zip(range(row, -1, -1), range(col, N)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens_util(board, row, N):
    if row == N:
        return True

    for col in range(N):
        if is_safe(board, row, col, N):
            board[row][col] = 1

            if solve_n_queens_util(board, row + 1, N):
                return True

            # If placing queen at board[row][col] doesn't lead to a solution,
            # then remove queen from board[row][col]
            board[row][col] = 0

    return False

def solve_n_queens(N):
    # Initialize the board
    board = [[0] * N for _ in range(N)]

    if not solve_n_queens_util(board, 0, N):
        print("Solution does not exist")
        return False

    # Print the solution
    print_solution(board)
    return True

def print_solution(board):
    for row in board:
        print(" ".join(map(str, row)))

# Example usage:
N = 8
solve_n_queens(N)

Q))))CAMEL BANANA PROBLEM
total_bananas = int(input("No. Of bananas at start : "))
distance = int(input("Distance to be covered  : "))
load_capacity = int(input("Maximum No. of bananas camel can carry at a time : "))

bananas_lost = 0
start = total_bananas
for i in range(distance) :
    while start > 0 :
        start = start-load_capacity
        if start == 1 :
            bananas_lost = bananas_lost-1
        bananas_lost = bananas_lost+2
    bananas_lost = bananas_lost-1
    start = total_bananas - bananas_lost
    if start == 0:
        break
print("Total bananas delivered : ", start )

Q)))WATER JUG PROBLEM
def water_jug_problem(jug1_cap, jug2_cap, target_amount):
    # Initialize the jugs and the possible actions
    j1 = 0
    j2 = 0
    actions = [("fill", 1), ("fill", 2), ("empty", 1), ("empty", 2), ("pour", 1, 2), ("pour", 2, 1)]
    # Create an empty set to store visited states
    visited = set()
    # Create a queue to store states to visit
    queue = [(j1, j2, [])]
    while queue:
        # Dequeue the front state from the queue
        j1, j2, seq = queue.pop(0)
        # If this state has not been visited before, mark it as visited
        if (j1, j2) not in visited:
            visited.add((j1, j2))
            # If this state matches the target amount, return the sequence of actions taken to get to this state
            if j1 == target_amount:
                return seq
            # Generate all possible next states from this state
            for action in actions:
                if action[0] == "fill":
                    if action[1] == 1:
                        next_state = (jug1_cap, j2)
                    else:
                        next_state = (j1, jug2_cap)
                elif action[0] == "empty":
                    if action[1] == 1:
                        next_state = (0, j2)
                    else:
                        next_state = (j1, 0)
                else:
                    if action[1] == 1:
                        amount = min(j1, jug2_cap - j2)
                        next_state = (j1 - amount, j2 + amount)
                    else:
                        amount = min(j2, jug1_cap - j1)
                        next_state = (j1 + amount, j2 - amount)
                # Add the next state to the queue if it has not been visited before
                if next_state not in visited:
                    next_seq = seq + [action]
                    queue.append((next_state[0], next_state[1], next_seq))
    # If the queue becomes empty without finding a solution, return None
    return None

result = water_jug_problem(5, 3, 1)
print(result)

Q)))TICTACTOE PROBLEM
# Set up the game board as a list
board = ["-", "-", "-",
        "-", "-", "-",
        "-", "-", "-"]

# Define a function to print the game board
def print_board():
    print(board[0] + " | " + board[1] + " | " + board[2])
    print(board[3] + " | " + board[4] + " | " + board[5])
    print(board[6] + " | " + board[7] + " | " + board[8])

# Define a function to handle a player's turn
def take_turn(player):
    print(player + "'s turn.")
    position = input("Choose a position from 1-9: ")
    while position not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        position = input("Invalid input. Choose a position from 1-9: ")
    position = int(position) - 1
    while board[position] != "-":
        position = int(input("Position already taken. Choose a different position: ")) - 1
    board[position] = player
    print_board()

# Define a function to check if the game is over
def check_game_over():
    # Check for a win
    if (board[0] == board[1] == board[2] != "-") or \
    (board[3] == board[4] == board[5] != "-") or \
    (board[6] == board[7] == board[8] != "-") or \
    (board[0] == board[3] == board[6] != "-") or \
    (board[1] == board[4] == board[7] != "-") or \
    (board[2] == board[5] == board[8] != "-") or \
    (board[0] == board[4] == board[8] != "-") or \
    (board[2] == board[4] == board[6] != "-"):
        return "win"
    # Check for a tie
    elif "-" not in board:
        return "tie"
    # Game is not over
    else:
        return "play"

# Define the main game loop
def play_game():
    print_board()
    current_player = "X"
    game_over = False
    while not game_over:
        take_turn(current_player)
        game_result = check_game_over()
        if game_result == "win":
            print(current_player + " wins!")
            game_over = True
        elif game_result == "tie":
            print("It's a tie!")
            game_over = True
        else:
            # Switch to the other player
            current_player = "O" if current_player == "X" else "X"

# Start the game
play_game()

Q)))CRIPTHARMATIC PROBLEM
from itertools import combinations, permutations
def replacements():
    for comb in combinations(range(10), 8):
        for perm in permutations(comb):
            if perm[0] * perm[1] != 0:
                yield dict(zip('BASELGMS', perm))
a, b, c = 'BASE', 'BALL', 'GAMES'
for replacement in replacements():
    f = lambda x: sum(replacement[e] * 10**i for i, e in enumerate(x[::-1]))
    if f(a) + f(b) == f(c):
        print('{} + {} = {}'.format(f(a), f(b), f(c)))

Q)))KMEANS
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


x = [1,2,2,1,8,9,8,9]
y = [1,2,1,2,8,9,9,8]

plt.scatter(x,y)
plt.show()


data = list(zip(x,y))
inertias = []

for i in range(1,8):
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(data)
  inertias.append(kmeans.inertia_)

plt.plot(range(1,8),inertias,marker="o")
plt.show()


kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
plt.scatter(x,y,c=kmeans.labels_)
plt.show()

Q)))KNN
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier



x = [1,2,2,1,8,9,8,9]
y = [1,2,1,2,8,9,9,8]
classes = [0,0,0,0,1,1,1,1]
plt.scatter(x,y,c=classes)
plt.show()

data = list(zip(x,y))
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(data,classes)

new_x = 8.5
new_y = 8.5

new_point = [(new_x,new_y)]
prediction = knn.predict(new_point)
print(prediction)

Q)))GRAPH COLOURING
V = 4

def print_solution(color):
    print("Solution Exists: Following are the assigned colors")
    print(" ".join(map(str, color)))

def is_safe(v, graph, color, c):
    # Check if the color 'c' is safe for the vertex 'v'
    for i in range(V):
        if graph[v][i] and c == color[i]:
            return False
    return True

def graph_coloring_util(graph, m, color, v):
    # Base case: If all vertices are assigned a color, return true
    if v == V:
        return True

    # Try different colors for the current vertex 'v'
    for c in range(1, m + 1):
        # Check if assignment of color 'c' to 'v' is fine
        if is_safe(v, graph, color, c):
            color[v] = c

            # Recur to assign colors to the rest of the vertices
            if graph_coloring_util(graph, m, color, v + 1):
                return True

            # If assigning color 'c' doesn't lead to a solution, remove it
            color[v] = 0

    # If no color can be assigned to this vertex, return false
    return False

def graph_coloring(graph, m):
    color = [0] * V

    # Call graph_coloring_util() for vertex 0
    if not graph_coloring_util(graph, m, color, 0):
        print("Solution does not exist")
        return False

    print_solution(color)
    return True

# Driver code
if _name_ == "_main_":
    graph = [
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0],
    ]

    m = 3
    graph_coloring(graph, m)
        

//////////////simple reflexive agent////////////////
class SimpleReflexAgent:
    def _init_(self):
        self.location = "A"

    def perceive(self, location):
        self.location = location

    def act(self):
        if self.location == "A":
            return "Right"
        elif self.location == "B":
            return "Left"
        elif self.location == "C":
            return "Up"
        elif self.location == "D":
            return "Down"
        else:
            return "No valid action"

# Example usage:
agent = SimpleReflexAgent()
print("Agent's current location:", agent.location)
print("Agent's action based on current percept:", agent.act())

# Suppose agent moves to location C
agent.perceive("C")
print("Agent's current location:", agent.location)
print("Agent's action based on current percept:", agent.act())
/////////////////Csp nqueens////////////
from constraint import Problem, AllDifferentConstraint

def nqueens(N):
    problem = Problem()

    # Define variables
    for i in range(N):
        problem.addVariable(i, range(N))

    # Define constraints
    problem.addConstraint(AllDifferentConstraint())

    for i in range(N):
        for j in range(i+1, N):
            problem.addConstraint(lambda x, y, i=i, j=j: abs(x-i) != abs(y-j) and x != y, (i, j))

    # Solve the problem
    solutions = problem.getSolutions()
    return solutions

# Example usage:
N = 8
solutions = nqueens(N)
print("Number of solutions:", len(solutions))
for solution in solutions:
    print(solution)
/////////////////////////////breadth fs/////////////
from collections import deque

def bfs(graph, start, goal):
    # Initialize the queue with the start node
    queue = deque([start])
    # Keep track of visited nodes
    visited = set([start])
    # Keep track of the path
    path = {start: None}

    while queue:
        node = queue.popleft()
        if node == goal:
            return construct_path(start, goal, path)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                path[neighbor] = node

    # If goal not found
    return None

def construct_path(start, goal, path):
    # Reconstruct the path from goal to start
    current = goal
    path_list = []
    while current != start:
        path_list.append(current)
        current = path[current]
    path_list.append(start)
    path_list.reverse()
    return path_list

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start_node = 'A'
goal_node = 'F'

result = bfs(graph, start_node, goal_node)
if result:
    print("Path found:", result)
else:
    print("No path found")
///////////////////dfs////////////////////////
def dfs(graph, start, goal):
    # Initialize stack with start node
    stack = [start]
    # Keep track of visited nodes
    visited = set()
    # Keep track of the path
    path = {start: None}

    while stack:
        node = stack.pop()
        if node == goal:
            return construct_path(start, goal, path)
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
                    path[neighbor] = node

    # If goal not found
    return None

def construct_path(start, goal, path):
    # Reconstruct the path from goal to start
    current = goal
    path_list = []
    while current != start:
        path_list.append(current)
        current = path[current]
    path_list.append(start)
    path_list.reverse()
    return path_list

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start_node = 'A'
goal_node = 'F'

result = dfs(graph, start_node, goal_node)
if result:
    print("Path found:", result)
else:
    print("No path found")
//////////////////////best fs////////////////////
from queue import PriorityQueue

v = 14

graph = [[] for i in range(v)]

def best_first_search(source,destination,graph):
  pq = PriorityQueue()
  visited = [False] * 14
  visited[source] = True

  pq.put((0,source))

  while pq:
    node = pq.get()[1]
    print(node)
    if node==destination:
      break

    for v,c in graph[node]:
      if visited[v] ==False:
        visited[v] = True
        pq.put((c,v))

def addedge(x,y,cost):
  graph[x].append((y,cost))
  graph[y].append((x,cost))

addedge(0, 1, 3)
addedge(0, 2, 6)
addedge(0, 3, 5)
addedge(1, 4, 9)
addedge(1, 5, 8)
addedge(2, 6, 12)
addedge(2, 7, 14)
addedge(3, 8, 7)
addedge(8, 9, 5)
addedge(8, 10, 6)
addedge(9, 11, 1)
addedge(9, 12, 10)
addedge(9, 13, 2)

source = 0
destination = 9

best_first_search(source,destination,graph)
///////////////////////////a star/////////////////
from queue import PriorityQueue
class PuzzleState:
    def _init_(self, puzzle, parent=None, move="Initial", cost=0):
        self.puzzle = puzzle
        self.parent = parent
        self.move = move
        self.cost = cost
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    def _eq_(self, other):
        return self.puzzle == other.puzzle

    def _lt_(self, other):
        return self.cost < other.cost

    def _hash_(self):
        return hash(str(self.puzzle))

    def h(self):
        return sum([1 if self.puzzle[i][j] != self.goal_state[i][j] else 0 for i in range(3) for j in range(3)])

    def get_successors(self):
        successors = []
        empty_row, empty_col = self.find_empty_tile()
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dr, dc in directions:
            new_row, new_col = empty_row + dr, empty_col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_puzzle = [row[:] for row in self.puzzle]
                new_puzzle[empty_row][empty_col], new_puzzle[new_row][new_col] = \
                    new_puzzle[new_row][new_col], new_puzzle[empty_row][empty_col]
                successors.append(PuzzleState(new_puzzle, self, "Move", self.cost + 1))

        return successors

    def find_empty_tile(self):
        for i in range(3):
            for j in range(3):
                if self.puzzle[i][j] == 0:
                    return i, j

def a_star_search(initial_state):
    frontier = PriorityQueue()
    frontier.put(initial_state)
    explored = set()

    while not frontier.empty():
        current_state = frontier.get()
        if current_state.puzzle == current_state.goal_state:
            return current_state

        explored.add(current_state)
        for successor in current_state.get_successors():
            if successor not in explored:
                frontier.put(successor)

    return None

def print_solution(solution):
    if solution is None:
        print("No solution found")
    else:
        path = []
        current_state = solution
        while current_state.parent:
            path.append((current_state.move, current_state.puzzle))
            current_state = current_state.parent
        path.append(("Initial", current_state.puzzle))

        path.reverse()
        for move, puzzle in path:
            print(move)
            print_puzzle(puzzle)

def print_puzzle(puzzle):
    for row in puzzle:
        print(row)
    print()

# Example usage:
initial_state = PuzzleState([[1, 2, 3], [4, 5, 6], [0, 7, 8]])
solution = a_star_search(initial_state)
print_solution(solution)
//////////////////////////////unification///////////////////////
def unify(x, y, theta):
    """
    Unify the two expressions x and y with the given substitution theta.
    """
    if theta is None:
        return None
    elif x == y:
        return theta
    elif isinstance(x, str) and x.islower():
        return unify_var(x, y, theta)
    elif isinstance(y, str) and y.islower():
        return unify_var(y, x, theta)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return None
        for xi, yi in zip(x, y):
            theta = unify(xi, yi, theta)
            if theta is None:
                return None
        return theta
    else:
        return None

def unify_var(var, x, theta):
    """
    Unify a variable var with expression x with the given substitution theta.
    """
    if var in theta:
        return unify(theta[var], x, theta)
    elif x in theta:
        return unify(var, theta[x], theta)
    else:
        theta[var] = x
        return theta

# Example usage:
theta = unify(['John', 'loves', 'Mary'], ['John', 'loves', 'Mary'], {})
print("Substitution:", theta)

theta = unify(['John', 'loves', 'Mary'], ['John', 'hates', 'Mary'], {})
print("Substitution:", theta)

theta = unify(['John', 'X', 'Y'], ['John', 'loves', 'Mary'], {'X': 'loves', 'Y': 'Mary'})
print("Substitution:", theta)
///////////////////////////////////////uncertain////////////
import random

def monty_hall_simulation(num_trials):
    switch_wins = 0
    stay_wins = 0

    for _ in range(num_trials):
        # print("")
        # print(f"{_} iteration ")
        doors = ['A', 'B', 'C']
        bike_location = random.choice(doors)
        # print("Bike location : ",bike_location)
        initial_choice = random.choice(doors)
        # print("player choice: ",initial_choice)
        doors.remove(initial_choice)

        if bike_location in doors:
            doors.remove(bike_location)
        monty_choice = random.choice(doors)

        # print("Monty's choice : ", monty_choice)
        doors = [d for d in ['A', 'B', 'C'] if d != monty_choice and d != initial_choice]
        final_choice = doors[0]

        stay_wins += (initial_choice == bike_location)
        switch_wins += (final_choice == bike_location)

    stay_win_prob = stay_wins / num_trials
    switch_win_prob = switch_wins / num_trials

    print(f"Probability of winning by staying: {stay_win_prob:.2f}")
    print(f"Probability of winning by switching: {switch_win_prob:.2f}")

# Number of trials
num_trials = 10

# Run simulation
monty_hall_simulation(num_trials)
////////////////////////learning algo////////////////
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Test the model
X_test = np.array([[6], [7], [8]])
predictions = model.predict(X_test)
print("Predictions:", predictions)
//////////////////////////////NLP/////////////////////////
import numpy as np

class Perceptron:
    def _init_(self, learning_rate=0.01, num_epochs=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)
        self.errors = []
        for _ in range(self.num_epochs):
            error = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                error += int(update != 0.0)
            self.errors.append(error)
        return self
    
    def predict(self, X):
        return np.where(np.dot(X, self.weights[1:]) + self.weights[0] >= 0.0, 1, -1)

# Example usage
X_train = np.array([[2, 3], [4, 5], [6, 7], [8, 9]])
y_train = np.array([1, -1, 1, -1])
model = Perceptron(learning_rate=0.1, num_epochs=10)
model.fit(X_train, y_train)
X_test = np.array([[1, 2], [5, 6]])
predictions = model.predict(X_test)
print("Predictions:", predictions)
//////////////////////////////////////DL DL////////////////////
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load and preprocess the dataset (e.g., MNIST)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data for CNN input
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
        `
      }],
      max_tokens:1500,
    },)
  };

  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', options);
    const data = await response.json();
    res.send(data);
  } catch (error) {
    console.log(error);
    res.status(500).send('Error fetching questions');
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
