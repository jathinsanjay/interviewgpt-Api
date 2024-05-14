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
def is_safe(board, row, col):
# Check if there is a queen in the same column
for i in range(row):
if board[i][col] == 1:
return False
# Check upper left diagonal
for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
if board[i][j] == 1:
return False
# Check upper right diagonal
for i, j in zip(range(row, -1, -1), range(col, len(board))):
if board[i][j] == 1:
return False
return True
def solve_n_queens(board, row):
n = len(board)
if row >= n:
return True
for col in range(n):
if is_safe(board, row, col):
board[row][col] = 1
if solve_n_queens(board, row + 1):
return True
board[row][col] = 0
return False
def print_solution(board):
for row in board:
print(" ".join(map(str, row)))
n = int(input("Enter the value of N: "))
board = [[0] * n for _ in range(n)]
if solve_n_queens(board, 0):
print("Solution:")
print_solution(board)
else:
print("No solution exists.")
//////////////simple reflexive agent////////////////
def simple_reflex_agent(percept):
location, status = percept
if status == "Dirty":
return "Suck"
elif location == "A":
return "Right"
elif location == "B":
return "Left"
# Test the agent with sample percepts
percept = ("A", "Dirty")
action = simple_reflex_agent(percept)
print("Percept:", percept)
print("Action:", action)
/////////////////Csp nqueens////////////
def is_safe(board, row, col, n):
for i in range(col):
if board[row][i] == 1:
return False
for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
if board[i][j] == 1:
return False
for i, j in zip(range(row, n, 1), range(col, -1, -1)):
if board[i][j] == 1:
return False
return True
def solve_n_queens_util(board, col, n):
if col >= n:
return True
for i in range(n):
if is_safe(board, i, col, n):
board[i][col] = 1
if solve_n_queens_util(board, col + 1, n):
return True
board[i][col] = 0
return False
def solve_n_queens(n):
board = [[0] * n for _ in range(n)]
if not solve_n_queens_util(board, 0, n):
print("No solution exists.")
return
for row in board:
print(" ".join(map(str, row)))
# Test the function
n = 4
solve_n_queens(n)
/////////////////////////////breadth fs/////////////
from collections import defaultdict, deque
class Graph:
def _init_(self):
self.graph = defaultdict(list)
def add_edge(self, u, v):
self.graph[u].append(v)
def bfs(self, start):
visited = set()
queue = deque([start])
visited.add(start)
while queue:
vertex = queue.popleft()
print(vertex, end=" ")
for neighbor in self.graph[vertex]:
if neighbor not in visited:
visited.add(neighbor)
queue.append(neighbor)
# Test the BFS function
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)
print("BFS Traversal starting from vertex 2:")
g.bfs(2)
///////////////////dfs////////////////////////
from collections import defaultdict
class Graph:
def _init_(self):
self.graph = defaultdict(list)
def add_edge(self, u, v):
self.graph[u].append(v)
def dfs_util(self, vertex, visited):
visited.add(vertex)
print(vertex, end=" ")
for neighbor in self.graph[vertex]:
if neighbor not in visited:
self.dfs_util(neighbor, visited)
def dfs(self, start):
visited = set()
self.dfs_util(start, visited)
# Test the DFS function
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)
print("DFS Traversal starting from vertex 2:")
g.dfs(2)
//////////////////////best fs////////////////////
from queue import PriorityQueue
class Graph:
def _init_(self):
self.graph = {}
def add_edge(self, u, v, weight):
if u not in self.graph:
self.graph[u] = []
self.graph[u].append((v, weight))
def best_first_search(self, start, goal):
visited = set()
pq = PriorityQueue()
pq.put((0, start))
while not pq.empty():
cost, current = pq.get()
visited.add(current)
print(current, end=" ")
if current == goal:
break
for neighbor, weight in self.graph.get(current, []):
if neighbor not in visited:
pq.put((weight, neighbor))
# Test the Best-First Search function
g = Graph()
g.add_edge('A', 'B', 5)
g.add_edge('A', 'C', 7)
g.add_edge('A', 'D', 9)
g.add_edge('B', 'E', 6)
g.add_edge('C', 'F', 10)
print("Best-First Search Traversal:")
g.best_first_search('A', 'F')
///////////////////////////a star/////////////////
from queue import PriorityQueue
class Graph:
def _init_(self):
self.graph = {}
def add_edge(self, u, v, weight):
if u not in self.graph:
self.graph[u] = []
self.graph[u].append((v, weight))
def astar_search(self, start, goal, heuristic):
visited = set()
pq = PriorityQueue()
pq.put((0, start))
while not pq.empty():
cost, current = pq.get()
visited.add(current)
print(current, end=" ")
if current == goal:
break
for neighbor, weight in self.graph.get(current, []):
if neighbor not in visited:
priority = cost + weight + heuristic(neighbor, goal)
pq.put((priority, neighbor))
# Test the A* Search function
g = Graph()
g.add_edge('A', 'B', 5)
g.add_edge('A', 'C', 7)
g.add_edge('A', 'D', 9)
g.add_edge('B', 'E', 6)
g.add_edge('C', 'F', 10)
def heuristic(node, goal):
# Example heuristic function (can be customized)
return 0
print("A* Search Traversal:")
g.astar_search('A', 'F', heuristic)
//////////////////////////////unification///////////////////////
def unify(var1, var2, theta):
if theta is None:
return None
elif var1 == var2:
return theta
elif isinstance(var1, str) and var1[0].islower():
return unify_var(var1, var2, theta)
elif isinstance(var2, str) and var2[0].islower():
return unify_var(var2, var1, theta)
elif isinstance(var1, list) and isinstance(var2, list):
if not var1 or not var2:
return unify(var1[1:], var2[1:], unify(var1[0], var2[0], theta))
else:
return unify(var1[1:], var2[1:], unify(var1[0], var2[0], theta))
else:
return None
def unify_var(var, x, theta):
if var in theta:
return unify(theta[var], x, theta)
elif x in theta:
return unify(var, theta[x], theta)
else:
theta[var] = x
return theta
# Test unification
print(unify('x', 'y', {})) # {'x': 'y'}
print(unify(['A', 'x'], ['A', 'y'], {})) # {'x': 'y'}
print(unify(['A', 'B', 'C'], ['A', 'B', 'C'], {})) # {}
///////////////////////////////////////uncertain////////////
import random
class MontyHallSimulation:
def _init_(self, num_trials):
self.num_trials = num_trials
def simulate(self):
switch_wins = 0
stay_wins = 0
for _ in range(self.num_trials):
# Randomly select a door with the car behind it
car_door = random.randint(1, 3)
# Contestant makes initial choice
initial_choice = random.randint(1, 3)
# Monty reveals a door with a goat behind it that the contestant didn't choose
remaining_doors = [door for door in range(1, 4) if door != initial_choice and door !=
car_door]
monty_reveals = random.choice(remaining_doors)
# Contestant decides whether to switch or stay
remaining_doors = [door for door in range(1, 4) if door != initial_choice and door !=
monty_reveals]
final_choice = initial_choice # Uncomment to stay with initial choice
# final_choice = remaining_doors[0] # Uncomment to switch doors
# Check if contestant wins
if final_choice == car_door:
if final_choice == initial_choice:
stay_wins += 1
else:
switch_wins += 1
stay_win_percentage = (stay_wins / self.num_trials) * 100
switch_win_percentage = (switch_wins / self.num_trials) * 100
print("Simulation results:")
print("Stay strategy wins: {:.2f}%".format(stay_win_percentage))
print("Switch strategy wins: {:.2f}%".format(switch_win_percentage))
# Example usage
num_trials = 10000
simulation = MontyHallSimulation(num_trials)
simulation.simulate()
////////////////////////learning algo////////////////
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
