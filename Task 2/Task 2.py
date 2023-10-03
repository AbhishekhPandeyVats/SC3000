import json
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

def dfs_with_energy_constraint(graph, start, target, energy_budget, dist_dict, cost_dict):
    stack = [(start, [start], 0, 0)]
    visited = set()

    while stack:
        node, path, accumulated_energy_cost, total_distance = stack.pop()
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
                stack.append((neighbor_node, new_path, new_accumulated_energy_cost, new_total_distance))

    return None, 0, 0

dfs_path, total_energy_cost, total_distance = dfs_with_energy_constraint(G, start_node, end_node, energy_budget, Dist, Cost)

if dfs_path:
    print("DFS Path:", "->".join(dfs_path))
    print("DFS Total Energy Cost:", total_energy_cost)
    print("DFS Total Distance:", total_distance)
else:
    print("DFS: No valid path found within the energy budget.")


def ucs_with_energy_constraint(graph, start, target, energy_budget, dist_dict, cost_dict):
    open_set = [(0, start, [start], 0, 0)]
    visited = set()

    while open_set:
        priority, node, path, accumulated_energy_cost, total_distance = heapq.heappop(open_set)
        if node in visited:
            continue

        visited.add(node)

        if node == target and accumulated_energy_cost <= energy_budget:
            return path, accumulated_energy_cost, total_distance

        neighbors = graph[node]
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
                heapq.heappush(open_set, (new_accumulated_energy_cost, neighbor_node, new_path, new_accumulated_energy_cost, new_total_distance))

    return None, 0, 0


ucs_path, total_energy_cost, total_distance = ucs_with_energy_constraint(G, start_node, end_node, energy_budget, Dist, Cost)


if ucs_path:
    print("UCS Path:", "->".join(ucs_path))
    print("UCS Total Energy Cost:", total_energy_cost)
    print("UCS Total Distance:", total_distance)
else:
    print("UCS: No valid path found within the energy budget.")