import json
#====================================================================================================================================================================================#
#Task 1

print("Task 1")

with open('G.json', 'r') as f:
    G = json.load(f)
with open('Coord.json', 'r') as f:
    Coord = json.load(f)
with open('Dist.json', 'r') as f:
    Dist = json.load(f)
with open('Cost.json', 'r') as f:
    Cost = json.load(f)
def uniform_cost_search(goal, start):

    # Initialize answer vector
    answer = [float('inf')] * len(goal)

    # Initialize a dictionary to store the path to each node
    paths = {node: [] for node in G.keys()}

    # Priority queue as a list of (cost, node, path) tuples
    queue = [(0, start, [])]

    # Create a set to store visited nodes
    visited = set()

    # Counter to keep track of the number of reached goals
    count = 0

    while queue:

        # Sort the queue by cost (ascending order)
        queue.sort()

        # Get the node with the lowest cost
        current_cost, current_node, current_path = queue.pop(0)

        # Check if the node is in the goal list
        if current_node in goal:
            index = goal.index(current_node)

            # If a new goal is reached
            if answer[index] == float('inf'):
                count += 1

            # If the cost is less
            if current_cost < answer[index]:
                answer[index] = current_cost

                # Update the path to this goal node
                paths[current_node] = current_path + [current_node]

            # If all goals are reached, return the answer and path
            if count == len(goal):
                return answer, paths

        # Check for non-visited nodes adjacent to the current node
        if current_node not in visited:
            for neighbor in G[current_node]:
                # Calculate the new cost for the neighbor
                edge_key = f"{current_node},{neighbor}"
                neighbor_cost = current_cost + Dist.get(edge_key, float('inf'))
                # Add neighbor to the queue with its cost and updated path
                queue.append((neighbor_cost, neighbor, current_path + [current_node]))

        # Mark the current node as visited
        visited.add(current_node)

    return answer, paths

# main function
if __name__ == '__main__':

    # goal state
    goal = []

    source = '1'
    target = '50'
    # set the goal
    # there can be multiple goal states
    goal.append(target)
    # Get the answer and paths
    answer, paths = uniform_cost_search(goal, source)

    # Print the minimum cost
    print("Minimum cost from ", source, " to", goal[0], " is = ", answer[0])

    # Print the shortest path
    goal_node = goal[0]
    shortest_path = paths[goal_node]
    print("Shortest path from ", source, " to", goal[0], " is: ")
    for i in range(0, len(shortest_path) -1):
        print(shortest_path[i], end=" -> ")
    print(shortest_path[-1])

#====================================================================================================================================================================================#
#Task 2
import heapq
with open('G.json', 'r') as f:
    G = json.load(f)
with open('Coord.json', 'r') as f:
    Coord = json.load(f)
with open('Dist.json', 'r') as f:
    Dist = json.load(f)
with open('Cost.json', 'r') as f:
    Cost = json.load(f)

energy_budget = 287932
start_node = '1'
end_node = '50'
print("Task 2")

def bfs_with_energy_constraint(graph, start, target, energy_budget, dist_dict, cost_dict):
    queue = [(start, [start], 0, 0)]
    visited = set()

    while queue:
        node, path, accumulated_energy_cost, total_distance = queue.pop(0)
        if node in visited:
            continue

        visited.add(node)

        if node == target and accumulated_energy_cost <= energy_budget:
            return path, accumulated_energy_cost, total_distance

        neighbors = graph[node]

        neighbors.sort(key=lambda neighbor: cost_dict.get(f"{node},{neighbor}", float('inf')))

        for neighbor_node in neighbors:
            edge_cost = cost_dict.get(f"{node},{neighbor_node}", 0)
            edge_distance = dist_dict.get(f"{node},{neighbor_node}", 0)
            new_accumulated_energy_cost = accumulated_energy_cost + edge_cost
            new_total_distance = total_distance + edge_distance
            new_path = path + [neighbor_node]

            if (
                    neighbor_node not in visited
                    and new_accumulated_energy_cost <= energy_budget
                    and new_path
            ):
                queue.append((neighbor_node, new_path, new_accumulated_energy_cost, new_total_distance))

    return None, 0, 0


bfs_path, total_energy_cost, total_distance = bfs_with_energy_constraint(G, start_node, end_node, energy_budget, Dist, Cost)

if bfs_path:
    print("BFS Path:", "->".join(bfs_path))
    print("BFS Total Energy Cost:", total_energy_cost)
    print("BFS Total Distance:", total_distance)
else:
    print("BFS: No valid path found within the energy budget.")

# def dfs_with_energy_constraint(graph, start, target, energy_budget, dist_dict, cost_dict):
#     stack = [(start, [start], 0, 0)]
#     visited = set()
#
#     while stack:
#         node, path, accumulated_energy_cost, total_distance = stack.pop()
#         if node in visited:
#             continue
#
#         visited.add(node)
#
#         if node == target and accumulated_energy_cost <= energy_budget:
#             return path, accumulated_energy_cost, total_distance
#
#         neighbors = graph[node]
#
#         neighbors.sort(key=lambda neighbor: cost_dict.get(f"{node},{neighbor}", float('inf')))
#
#         for neighbor_node in neighbors:
#             edge_cost = cost_dict.get(f"{node},{neighbor_node}", 0)
#             edge_distance = dist_dict.get(f"{node},{neighbor_node}", 0)
#             new_accumulated_energy_cost = accumulated_energy_cost + edge_cost
#             new_total_distance = total_distance + edge_distance
#             new_path = path + [neighbor_node]
#
#             if (
#                     neighbor_node not in visited
#                     and new_accumulated_energy_cost <= energy_budget
#                     and new_path
#             ):
#                 stack.append((neighbor_node, new_path, new_accumulated_energy_cost, new_total_distance))
#
#     return None, 0, 0
#
# dfs_path, total_energy_cost, total_distance = dfs_with_energy_constraint(G, start_node, end_node, energy_budget, Dist, Cost)
#
# if dfs_path:
#     print("DFS Path:", "->".join(dfs_path))
#     print("DFS Total Energy Cost:", total_energy_cost)
#     print("DFS Total Distance:", total_distance)
# else:
#     print("DFS: No valid path found within the energy budget.")
#
#
# def ucs_with_energy_constraint(graph, start, target, energy_budget, dist_dict, cost_dict):
#     open_set = [(0, start, [start], 0, 0)]
#     visited = set()
#
#     while open_set:
#         priority, node, path, accumulated_energy_cost, total_distance = heapq.heappop(open_set)
#         if node in visited:
#             continue
#
#         visited.add(node)
#
#         if node == target and accumulated_energy_cost <= energy_budget:
#             return path, accumulated_energy_cost, total_distance
#
#         neighbors = graph[node]
#         for neighbor_node in neighbors:
#             edge_cost = cost_dict.get(f"{node},{neighbor_node}", 0)
#             edge_distance = dist_dict.get(f"{node},{neighbor_node}", 0)
#             new_accumulated_energy_cost = accumulated_energy_cost + edge_cost
#             new_total_distance = total_distance + edge_distance
#             new_path = path + [neighbor_node]
#
#             if (
#                     neighbor_node not in visited
#                     and new_accumulated_energy_cost <= energy_budget
#                     and new_path
#             ):
#                 heapq.heappush(open_set, (new_accumulated_energy_cost, neighbor_node, new_path, new_accumulated_energy_cost, new_total_distance))
#
#     return None, 0, 0
#
#
# ucs_path, total_energy_cost, total_distance = ucs_with_energy_constraint(G, start_node, end_node, energy_budget, Dist, Cost)
#
#
# if ucs_path:
#     print("UCS Path:", "->".join(ucs_path))
#     print("UCS Total Energy Cost:", total_energy_cost)
#     print("UCS Total Distance:", total_distance)
# else:
#     print("UCS: No valid path found within the energy budget.")

#====================================================================================================================================================================================#
#Task 3

print("Task 3")

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
gamma = 1.0 # 1.0 is the default value


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