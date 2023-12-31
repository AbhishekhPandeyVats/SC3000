import json

with open('G.json', 'r') as f:
    G = json.load(f)
with open('Coord.json', 'r') as f:
    Coord = json.load(f)
with open('Dist.json', 'r') as f:
    Dist = json.load(f)
with open('Cost.json', 'r') as f:
    Cost = json.load(f)
#====================================================================================================================================================================================#

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

    source = input('Please enter the start node: ')
    target = input('Please enter the goal node: ')
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


