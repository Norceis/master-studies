import math
import random
import time
from matplotlib.pyplot import show, figure, savefig, plot
from collections import deque
from queue import PriorityQueue
import tracemalloc
import copy


CITIES = 12
ASYMMETRICAL = True
BROKEN_EDGES = True

class Graph:

    def __init__(self, vertices: list, edges: list):
        self.vertices = vertices
        self.edges = edges

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    def __getitem__(self, item):
        return self.vertices[item]

    def __str__(self):
        return "Graph(" + str(self.vertices) + " " + str(self.edges) + ")"

    def __hash__(self):
        return hash((self.vertices, self.edges))

    def __sub__(self, other):
        return [x for x in self.vertices if x not in other]

    def remove_edges(self, percent):
        number_of_removed_edges = math.floor(percent * len(self.edges))
        for edge in range(number_of_removed_edges):
            self.edges.remove(self.edges[random.randint(0, len(self.edges)-1)])

    def generate_connections(self, asymmetrical):
        for x in self.vertices:
            for y in self.vertices:
                if x is not y:
                    if asymmetrical:
                        if x.z > y.z:
                            self.edges.append(Edge(x, y, (calculate_distance(x, y) * 0.9)))
                        elif x.z < y.z:
                            self.edges.append(Edge(x, y, (calculate_distance(x, y) * 1.1)))
                        else:
                            self.edges.append(Edge(x, y, calculate_distance(x, y)))
                    else:
                        self.edges.append(Edge(x, y, calculate_distance(x, y)))

class Vertex:
    def __init__(self, coordinates: list):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.z = coordinates[2]

    def __iter__(self):
        return iter([self.x, self.y, self.z])

    def __repr__(self):
        return "Vertex(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

    def __str__(self):
        return "Vertex(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

    def __getitem__(self, item):
        return [self.x, self.y, self.z].__getitem__(item)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

class Edge:
    def __init__(self, source: Vertex, target: Vertex, cost: float, pheromone=1):
        self.source = source
        self.target = target
        self.cost = cost
        self.pheromone = pheromone

    def __iter__(self):
        return iter([self.source, self.target, self.cost])

    def __repr__(self):
        return "Edge(" + str(self.source) + ", " + str(self.target) + ", " + str(self.cost) + ")"

    def __str__(self):
        return "Edge(" + str(self.source) + ", " + str(self.target) + ", " + str(self.cost) + ")"

    def __getitem__(self, item):
        return [self.source, self.target, self.cost].__getitem__(item)

    def __hash__(self):
        return hash((self.source, self.target, self.cost))

class Road:
    def __init__(self, vertices, cost, approximate=None):
        self.vertices = vertices
        self.cost = cost
        self.approximate = approximate

    def add_cost(self, value):
        self.cost += value

    def cities_to_visit(self, all_vertices: list[Vertex]):
        return [x for x in all_vertices if x not in self.vertices]

    def __repr__(self):
        return "Road(" + str(self.vertices) + ", " + str(self.cost) + ")"

    def __str__(self):
        return "Road(" + str(self.vertices) + ", " + str(self.cost) + ")"

    def __lt__(self, other):
        if self.approximate is None:
            return self.cost < other.cost
        else:
            return self.approximate < other.approximate

    def __le__(self, other):
        if self.approximate is None:
            return self.cost <= other.cost
        else:
            return self.approximate <= other.approximate

    def __getitem__(self, item: int):
        if item == 0:
            if self.approximate is None:
                return self.cost
            else:
                return self.approximate
        elif item == 1:
            return self.vertices
        else:
            raise IndexError

def generate_city():
    return [random.randint(-100, 100), random.randint(-100, 100), random.randint(0, 50)]

def calculate_distance(x, y):
    return ((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2 + (y[2] - x[2]) ** 2) ** (1 / 2)

def show_graph_3d_plot(vertices_list, name, edges=None, path=None, show_Plot=True):
    xs, ys, zs = zip(*vertices_list)

    fig = figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(xs, ys, zs, s=100)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if edges and path is None:
        already_connected = list()
        for a in edges:
            if a[0] != a[1] and (a[0], a[1]) not in already_connected:
                plot([a[0][0], a[1][0]],
                     [a[0][1], a[1][1]],
                     [a[0][2], a[1][2]], '--')
                already_connected.append((a[0], a[1]))
                already_connected.append((a[1], a[0]))
    elif path is not None:
        for i in range(len(path) - 1):
            plot([path[i][0], path[i + 1][0]],
                 [path[i][1], path[i + 1][1]],
                 [path[i][2], path[i + 1][2]])
        pass

    if show_Plot:
        show()
    else:
        savefig(name)

def bfs(graph):
    roads = []

    start = graph.vertices[0]
    queue = deque()
    queue.append(Road([start], 0))

    while len(queue) > 0:
        popped_path = queue.popleft()
        last_node = popped_path.vertices[-1]

        if len(popped_path.vertices) is len(graph):
            if len(list(filter(lambda e: e.source is last_node and e.target is start, graph.edges))) != 0:
                last_edge = list(filter(lambda e: e.source is last_node and e.target is start, graph.edges))[0]
                popped_path.vertices.append(last_edge.target)
                popped_path.cost += last_edge.cost
                roads.append(popped_path)
        else:
            for edge in graph.edges:
                if edge.source is last_node and edge.target not in popped_path.vertices:
                    road_vertices_copy = copy.copy(popped_path.vertices)
                    road_vertices_copy.append(edge.target)
                    new_cost = popped_path.cost + edge.cost
                    queue.append(Road(road_vertices_copy, new_cost))

    path = min(roads, key=lambda x: x.cost)
    return path

def dfs(graph, current, visited=None, roads=None):

    if roads is None:
        roads = []
    if visited is None:
        visited = Road([], 0)

    visited.vertices.append(current)

    if len(visited.vertices) is len(graph):
        if len(list(filter(lambda e: e.source is current and e.target is visited.vertices[0], graph.edges))) != 0:
            last_edge = list(filter(lambda e: e.source is current and e.target is visited.vertices[0], graph.edges))[0]
            visited.vertices.append(last_edge.target)
            visited.cost += last_edge.cost
            roads.append(visited)
            return

    for edge in graph.edges:
        if edge.source == current and edge.target not in visited.vertices:
            dfs(graph, edge.target, Road(list(visited.vertices), visited.cost + edge.cost), roads)

    return roads

def greedy(graph):
    try:
        path = Road([], 0)
        random_first_vertex = random.randint(0, len(graph)-1)
        root = graph.vertices[random_first_vertex]
        path.vertices.append(root)

        for _ in graph.vertices:
            possible_edges = filter(lambda e: e.source is root, graph.edges)
            sorted_edges = sorted(possible_edges, key=lambda x: x[2])
            for edge in sorted_edges:
                if edge[1] not in path.vertices:
                    root = edge[1]
                    path.vertices.append(root)
                    break
                else:
                    pass

        path.vertices.append(graph.vertices[random_first_vertex])

        while len(graph) != len(path.vertices)-1:
            path = greedy(graph)

        try:
            for road in range(len(path.vertices)):
                if road == len(path.vertices)-1:
                    break
                path.cost += list(filter(lambda e: e.source is path.vertices[road] and e.target is path.vertices[road+1], graph.edges))[0][2]
        except:
            path = greedy_second_degree(graph)

        return path

    except:
        print(f'The road cannot be found (greedy)')

def greedy_second_degree(graph):
    try:
        path = Road([], 0)
        random_first_vertex = random.randint(0, len(graph)-1)
        root = graph.vertices[random_first_vertex]
        path.vertices.append(root)

        for _ in graph.vertices:
            possible_edges = filter(lambda e: e.source is root, graph.edges)
            edges_sum = []

            for edge in possible_edges:
                second_degree_possible_edges = filter(lambda e: e.source is edge[1] and e.target is not root, graph.edges)
                sorted_second_degree_edges = sorted(second_degree_possible_edges, key=lambda x: x[2])
                first_cost = list(filter(lambda e: e.source is root and e.target is edge[1], graph.edges))[0][2]
                second_cost = list(sorted_second_degree_edges)[0][2]
                edges_sum.append([root, edge[1], first_cost+second_cost])

            sorted_edges = sorted(edges_sum, key=lambda x: x[2])
            for edge in sorted_edges:
                if edge[1] not in path.vertices:
                    root = edge[1]
                    path.vertices.append(root)
                    break
                else:
                    pass
        path.vertices.append(graph.vertices[random_first_vertex])

        while len(graph) != len(path.vertices)-1:
            path = greedy_second_degree(graph)

        try:
            for road in range(len(path.vertices)):
                if road == len(path.vertices)-1:
                    break
                path.cost += list(filter(lambda e: e.source is path.vertices[road] and e.target is path.vertices[road+1], graph.edges))[0][2]
        except:
            path = greedy_second_degree(graph)

        return path

    except:
        print(f'The road cannot be found (second degree greedy)')

def a_star(graph, heuristic):
    queue = PriorityQueue()
    start = graph.vertices[0]
    queue.put(Road([start], 0, 0))
    if heuristic == 'min':
        minimal_cost = min(graph.edges, key=lambda edge: edge.cost).cost
    elif heuristic == 'max':
        minimal_cost = max(graph.edges, key=lambda edge: edge.cost).cost

    while queue.qsize() > 0:
        min_road = queue.get()
        possible_outputs = min_road.cities_to_visit(graph.vertices)

        if BROKEN_EDGES:
            filtered_outputs = []
            for possible_output in possible_outputs:
                if len(list(filter(lambda e: e.source is min_road.vertices[-1] and e.target is possible_output, graph.edges))) != 0:
                    filtered_outputs.append(possible_output)
            possible_outputs = list(filtered_outputs)

        if len(min_road.vertices) == len(graph) + 1:
            return min_road

        if len(min_road.vertices) == len(graph):
            if len(possible_outputs) == 0:
                if min_road.vertices[0] != min_road.vertices[-1]:
                    if len(list(filter(lambda e: e.source is min_road.vertices[-1] and e.target is start, graph.edges))) != 0:
                        possible_outputs.append(start)

        for next in possible_outputs:
            min_road_vertices_copy = copy.copy(min_road.vertices)
            min_road_vertices_copy.append(next)
            basic_cost = min_road.cost + list(filter(lambda e: e.source is min_road_vertices_copy[-2] and e.target is min_road_vertices_copy[-1], graph.edges))[0][2]
            heuristic_cost = (1 + len(graph) - len(min_road_vertices_copy)) * minimal_cost
            queue.put(Road(min_road_vertices_copy, basic_cost, basic_cost + heuristic_cost))

def ants(graph, epochs=100, ants=CITIES, alpha=1, beta=1, Q=2, ro=0.1):

    best_ant = Road([], 100000)

    for epoch in range(epochs):
        for ant_number in range(ants):
            start = graph.vertices[random.randint(0, len(graph)-1)]
            ant = Road([start], 0)
            possible_outputs = ant.cities_to_visit(graph.vertices)
            while len(ant.vertices) != len(graph)+1:

                if ant.vertices[0] != ant.vertices[-1]:
                    possible_outputs = ant.cities_to_visit(graph.vertices)

                if len(ant.vertices) == len(graph):
                    if ant.vertices[0] != ant.vertices[-1]:
                        if len(list(filter(lambda e: e.source is ant.vertices[-1] and e.target is start,
                                           graph.edges))) != 0:
                            possible_outputs.append(start)
                        else:
                            break

                if BROKEN_EDGES:
                    filtered_outputs = []
                    for possible_output in possible_outputs:
                        if len(list(filter(lambda e: e.source is ant.vertices[-1] and e.target is possible_output,
                                           graph.edges))) != 0:
                            filtered_outputs.append(possible_output)
                    possible_outputs = list(filtered_outputs)
                    if len(possible_outputs) == 0:
                        break

                dictionary_of_costs = {}
                for next in possible_outputs:
                    next_edge = list(filter(lambda e: e.source is ant.vertices[-1] and e.target is next, graph.edges))[0]
                    dictionary_of_costs[next_edge] = (next_edge.pheromone ** alpha * (1/next_edge.cost) ** beta) / \
                                                     sum([other_edge.pheromone ** alpha * 1/other_edge.cost **
                                                          beta for other_edge in
                                                          list(filter(lambda e: e.source is ant.vertices[-1], graph.edges))])

                chosen_edge = random.choices(list(dictionary_of_costs.keys()), weights=list(dictionary_of_costs.values()))
                ant.vertices.append(chosen_edge[-1].target)
                ant.add_cost(chosen_edge[-1].cost)
                chosen_edge[-1].pheromone += Q

            if ant.cost < best_ant.cost and len(ant.vertices) == (len(graph) + 1):
                best_ant = ant
            for edge in graph.edges:
                edge.pheromone *= (1-ro)

    return best_ant

my_vertices_list = []
for i in range(CITIES):
    my_vertices_list.append(Vertex(generate_city()))

graf = Graph(my_vertices_list, [])
graf.generate_connections(True)

if BROKEN_EDGES:
    graf.remove_edges(0.2)

# path_bfs = bfs(graf)
# path_dfs = min(dfs(graf, graf[0]), key=lambda x: x.cost)
# path_greedy = greedy(graf)
# path_second_degree_greedy = greedy_second_degree(graf)
# path_a_star_min = a_star(graf, 'min')
# path_a_star_max = a_star(graf, 'max')
# path_ants = ants(graf)

tracemalloc.start()
start_bfs = time.perf_counter()
path_bfs = bfs(graf)
current, peak = tracemalloc.get_traced_memory()
print(f"Cost of path found by BFS algorithm is: {round(path_bfs.cost)}\n"
      f"BFS time: {round((time.perf_counter() - start_bfs), 4)}\n"
      f"BFS peak memory usage was {round((peak / 10**6), 4)} MB\n")
tracemalloc.stop()

tracemalloc.start()
start_dfs = time.perf_counter()
path_dfs = min(dfs(graf, graf[0]), key=lambda x: x.cost)
_, peak = tracemalloc.get_traced_memory()
print(f"Cost of path found by DFS algorithm is: {round(path_dfs.cost)}\n"
      f"DFS time: {round((time.perf_counter() - start_dfs), 4)}\n"
      f"DFS peak memory usage was {round((peak / 10**6), 4)} MB\n")
tracemalloc.stop()

tracemalloc.start()
start_greedy = time.perf_counter()
path_greedy = greedy(graf)
_, peak = tracemalloc.get_traced_memory()
print(f"Cost of path found by greedy algorithm is: {round(path_greedy.cost)}\n"
      f"Greedy time: {round((time.perf_counter() - start_greedy), 4)}\n"
      f"Greedy peak memory usage was {round((peak / 10**6), 4)} MB\n")
tracemalloc.stop()

tracemalloc.start()
start_greedy_second_degree = time.perf_counter()
path_greedy_second_degree = greedy_second_degree(graf)
_, peak = tracemalloc.get_traced_memory()
print(f"Cost of path found by greedy_second_degree algorithm is: {round(path_greedy_second_degree.cost)}\n"
      f"Greedy_second_degree time: {round((time.perf_counter() - start_greedy_second_degree), 4)}\n"
      f"Greedy_second_degree peak memory usage was {round((peak / 10**6), 4)} MB\n")
tracemalloc.stop()

tracemalloc.start()
start_a_star_min = time.perf_counter()
path_a_star_min = a_star(graf, 'min')
_, peak = tracemalloc.get_traced_memory()
print(f"Cost of path found by a_star_min algorithm is: {round(path_a_star_min.cost)}\n"
      f"A_star_min time: {round((time.perf_counter() - start_a_star_min), 4)}\n"
      f"A_star_min peak memory usage was {round((peak / 10**6), 4)} MB\n")
tracemalloc.stop()

tracemalloc.start()
start_a_star_max = time.perf_counter()
path_a_star_max = a_star(graf, 'max')
_, peak = tracemalloc.get_traced_memory()
print(f"Cost of path found by a_star_max algorithm is: {round(path_a_star_max.cost)}\n"
      f"A_star_max time: {round((time.perf_counter() - start_a_star_max), 4)}\n"
      f"A_star_max peak memory usage was {round((peak / 10**6), 4)} MB\n")
tracemalloc.stop()

tracemalloc.start()
start_ants = time.perf_counter()
path_ants = ants(graf)
_, peak = tracemalloc.get_traced_memory()
print(f"Cost of path found by ants algorithm is: {round(path_ants.cost)}\n"
      f"Ants time: {round((time.perf_counter() - start_ants), 4)}\n"
      f"Ants peak memory usage was {round((peak / 10**6), 4)} MB\n")
tracemalloc.stop()

# print(path_dfs.cost)
# print(path_bfs.cost)
# print(path_greedy.cost)
# print(path_greedy_second_degree.cost)
# print(path_a_star_min.cost)
# print(path_a_star_max.cost)
# print(path_ants.cost)

show_graph_3d_plot(my_vertices_list, path=path_ants.vertices, name='', show_Plot=True)