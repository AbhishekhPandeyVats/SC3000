# Task 3: A* Star Search

# We will use Python's Heap Queue as a Priority Queue
import heapq 
import json
import math
import time
import psutil

# Reading the JSON Files
with open('G.json', 'r') as f:
    graph_data = json.load(f)

with open('Coord.json', 'r') as f:
    coord_data = json.load(f)

with open('Dist.json', 'r') as f:
    dist_data = json.load(f)

with open('Cost.json', 'r') as f:
    cost_data = json.load(f)


# Input data
energy_budget = 287932
start_node = "1"
goal_node = "50"
gamma = 0.2 # 1.0 is the default value


# Function to calculate Euclidean Distance, which will be used for our Shortest Distance output
def euclideanDistance(node1, node2, Coord, gamma):

    # retrieving the coordinates (X, Y) for both nodes
    x1, y1 = Coord[node1]
    x2, y2 = Coord[node2]

    # calculating the Euclidean distance with Gamma multiplier
    distance = gamma * math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  

    return distance


# A* Search Function
def astarSearch(G, Coord, Dist, Cost, start, goal, gamma):

    # initialise Priority Queue for open nodes
    open_list = []  

    # push in the Start Node
    heapq.heappush(open_list, (0, start)) 

    # dictionary to store Parent pointers
    came_from = {}  
    
    # accumulated Cost Dictionary
    g_score = {node: float('inf') for node in G}  
    g_score[start] = 0

    # initialise total distance travelled
    total_distance = 0

    # initialise total energy cost to keep track of our energy budget
    total_energy_cost = 0



    while open_list:

        # get the Node with lowest f-score, where f-score is the sum of g-score (cost incurred so far) and h-score (heuristic cost estimate)
        _, current = heapq.heappop(open_list) 

        if current == goal:
            path = reconstructPath(came_from, current)
            energy_cost = calculateEnergyCost(path,Cost)
            return path, energy_cost, total_distance

        for neighbour in G[current]:

            # calculating tentative g_score
            tentative_g_score = g_score[current] + Dist[f"{current},{neighbour}"]

            # check if this path exceeds the energy budget, if so, skip it
            if total_energy_cost + tentative_g_score > energy_budget:
                continue

            # if this condition is True, it means the current path to neighbour is better than any other previous paths, so we should consider this path
            if tentative_g_score < g_score[neighbour]:

                # update the came_from dictionary as well as the new lower g-score
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g_score

                # calculate f-score
                f_score = g_score[neighbour] + euclideanDistance(neighbour, goal, Coord, gamma)

                # pushes the neighbour to the open_list Priority Queue with its corresponding f-score, meaning the nodes with lower f-score will be explored first
                heapq.heappush(open_list, (f_score, neighbour))

                # update total distance travelled along the current path
                total_distance += Dist[f"{current},{neighbour}"]

    return "No Path Found", 0, 0


# Function to reconstruct most optimal path from the came_from dictionary
def reconstructPath(came_from, current):
    path = []
    while current in came_from:
        path.insert(0, current)
        current = came_from[current]

    # Add the start node to the path
    path.insert(0, current)  

    formatted_path = "->".join(map(str,path))
    return formatted_path


# Function to calculate total Energy Cost for the path
def calculateEnergyCost(path, Cost):
    energy_cost = 0
    for i in range(len(path) -1):
        edge = f"{path[i]},{path[i+1]}"
        energy_cost += Cost.get(edge,0) 

    return energy_cost

# Function to measure Memory Usage
process = psutil.Process()
memory_usage = process.memory_info().rss 



# Performance Measurement
def measure_performance(graph_data, coord_data, dist_data, cost_data, start_node, goal_node, gamma):
    start_time = time.time()
    path, energy_cost, total_distance = astarSearch(graph_data, coord_data, dist_data, cost_data, start_node, goal_node, gamma)
    end_time = time.time()
    runtime = end_time - start_time

    if path != "No Path Found":
        print("Shortest path: S->" + path + "->T")
        print("Shortest distance:", total_distance)
        print("Energy Cost:", energy_cost)
        print("Runtime: {:.6f} seconds".format(runtime))
        print("Memory Usage: {} bytes".format(memory_usage))
    else:
        print("No Path Found!")


'''
# Iterating A* Search n times 

iterations = 1
runtimes = []
memory_usages = []

for _ in range(iterations):
    start_time = time.time()
    path, energy_cost, total_distance = astarSearch(graph_data, coord_data, dist_data, cost_data, start_node, goal_node, gamma)
    end_time = time.time()
    runtime = end_time - start_time
    runtimes.append(runtime)
    memory_usage = process.memory_info().rss
    memory_usages.append(memory_usage)

total_runtime = sum(runtimes)
average_runtime = sum(runtimes)/iterations
total_memory_usage = sum(memory_usages)
average_memory_usage = sum(memory_usages)/iterations
print("Average Runtime over {} iterations: {:.6f} seconds".format(iterations, average_runtime))
print("Total Runtime over {} iterations: {:.6f} seconds".format(iterations, total_runtime))
print("Average Memory Usage over {} iterations: {} bytes".format(iterations, average_memory_usage))
print("Total Memory Usage over {} iterations: {} bytes".format(iterations, total_memory_usage))
'''


# Task 3 Output
measure_performance(graph_data, coord_data, dist_data, cost_data, start_node, goal_node, gamma)
