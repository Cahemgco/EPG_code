import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy
import networkx as nx
import random

from graph import Graph, a_star
from parameter import *
from copy import deepcopy
import os

if not train_mode:
    from test_parameter import *

class Graph_generator:
    def __init__(self, map_size, k_size, plot=False, test=True):
        self.k_size = k_size
        self.test = test
        self.graph = Graph()
        self.node_coords = None
        self.plot = plot
        self.max_dist = 0
        self.x = []
        self.y = []
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.dist_threshold = (((self.map_x-1)/(NUMX-1))**2 + ((self.map_y-1)/(NUMY-1))**2)**0.5 * 1.1
        self.uniform_points = self.generate_uniform_points()
        self.nodes_list = []
        self.reference_policy = None
        self.adjacent_matrix = []
        self.network_adjacent_matrix = []

    def generate_graph(self, robot_belief, random_seed):
        # get node_coords by finding the uniform points in free area
        free_area = self.free_area(robot_belief)
        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        node_coords = self.uniform_points[candidate_indices]
        
        self.node_coords = self.unique_coords(node_coords).reshape(-1, 2)
        node_num = len(self.node_coords)

        # generate the collision free graph
        self.find_k_neighbor_all_nodes(self.node_coords, robot_belief)
        # self.dgl_graph = dgl.graph((self.npedges[0],self.npedges[1]), num_nodes=len(self.node_coords))
        ## generate reference policy
        graph = list(self.graph.edges.values())
        edge_inputs = []
        for i, node in enumerate(graph):
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)
            assert i == node_edges[0]
        
        ## generate adjacent matrix
        self.adjacent_matrix = self.calculate_edge_mask(edge_inputs)

        dist = [[9999 for i in range(len(self.adjacent_matrix))] for j in range(len(self.adjacent_matrix))]
        next_node = [[j for j in range(node_num)] for i in range(node_num)]

        for i in range(len(self.adjacent_matrix)):
            for j in range(len(self.adjacent_matrix)):
                if i == j:
                    dist[i][j] = 0
                elif self.adjacent_matrix[i][j] == 0:
                    dist[i][j] = 1
        self.max_dist = np.max(dist)

        for k in range(len(self.adjacent_matrix)):
            for i in range(len(self.adjacent_matrix)):
                for j in range(len(self.adjacent_matrix)):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]

        self.network_adjacent_matrix = np.array(dist)
        self.next_node = np.array(next_node)

        # calculate the feature as the number of observable frontiers of each node
        # save the observable frontiers to be reused
        adj_matrix = 1 - self.adjacent_matrix
        graph = nx.from_numpy_array(adj_matrix)

        if EXIT_NUM > 0:   
            # if self.test == True:   
            while True:
                # random.seed(42)
                self.exit_indexes = random.sample(range(node_num), EXIT_NUM)

                # random.seed() 
                candidates = [index for index in range(node_num) if index not in self.exit_indexes]
                robot_indexes = random.sample(candidates, N_ROBOTS)
                robot_positions = [self.node_coords[robot] for robot in robot_indexes]

                min_dist = MIN_EVADOR_EXIT_DIST
                valid_target_candidates = [
                    node for node in candidates
                    if any(
                        self.network_adjacent_matrix[node][exit_index] == min_dist
                        for exit_index in self.exit_indexes
                    )
                ]

                if not valid_target_candidates:
                    print("No valid target nodes found for the desired distance. Adjust parameters.")
                    continue

                target_index = random.choice(valid_target_candidates)
                target_position = self.node_coords[target_index]

                closest_exit_index = min(
                    self.exit_indexes,
                    key=lambda exit_index: self.network_adjacent_matrix[target_index][exit_index]
                )
                closest_exit_distance = self.network_adjacent_matrix[target_index][closest_exit_index]
    
                if closest_exit_distance != min_dist:
                    continue

                valid_exits = [
                    exit_index for exit_index in self.exit_indexes
                    if self.network_adjacent_matrix[target_index][exit_index] <= 10
                ]

                all_exits_satisfied = True
                for exit_index in valid_exits:
                    closer_robots = [
                        robot_index for robot_index in robot_indexes
                        if self.network_adjacent_matrix[robot_index][exit_index] < self.network_adjacent_matrix[target_index][exit_index]
                    ]
                    if not closer_robots:
                        all_exits_satisfied = False
                        break

                if not all_exits_satisfied:
                    continue

                break
                
        else:
            raise ValueError("Must Have Exit")
      
        node_feature = []
        for index in range(len(self.node_coords)):
            feature = []
            feature.append(self.network_adjacent_matrix[index][target_index])
            for robot in range(len(robot_indexes)):
                feature.append(self.network_adjacent_matrix[index][robot_indexes[robot]])
            if EXIT_NUM > 0:
                for exit in self.exit_indexes:
                    feature.append(self.network_adjacent_matrix[index][exit])
            # print(index, feature)
            node_feature.append(feature)
        self.node_feature = np.array(node_feature)
        # print('node_feature', self.node_feature.shape)

        if EXIT_NUM > 0:
            return robot_positions, robot_indexes, target_position, target_index, self.node_coords, self.graph.edges, self.node_feature, robot_indexes, target_index, self.adjacent_matrix, self.network_adjacent_matrix, self.next_node, self.exit_indexes
        else:
            return robot_positions, robot_indexes, target_position, target_index, self.node_coords, self.graph.edges, self.node_feature, robot_indexes, target_index, self.adjacent_matrix, self.network_adjacent_matrix, self.next_node

    def calculate_edge_mask(self, edge_inputs):
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            cnt = 0
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
                    cnt += 1
            assert cnt <= K_SIZE
        return bias_matrix

    def update_graph(self, robot_indexes, target_index):
        node_feature = []
        for index in range(len(self.node_coords)):
            feature = []
            feature.append(self.network_adjacent_matrix[index][target_index])
            for robot in range(len(robot_indexes)):
                feature.append(self.network_adjacent_matrix[index][robot_indexes[robot]])
            # print(index, feature)
            if EXIT_NUM > 0:
                for exit in self.exit_indexes:
                    feature.append(self.network_adjacent_matrix[index][exit])
            node_feature.append(feature)
        self.node_feature = np.array(node_feature)

        return self.node_feature

    def generate_uniform_points(self):
        x = np.linspace(0, self.map_x - 1, NUMX).round().astype(int) # 55
        y = np.linspace(0, self.map_y - 1, NUMY).round().astype(int)


        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def free_area(self, robot_belief):
        index = np.where(robot_belief == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def unique_coords(self, coords):
        x = coords[:, 0] + coords[:, 1] * 1j
        indices = np.unique(x, return_index=True)[1]
        coords = np.array([coords[idx] for idx in sorted(indices)])
        return coords

    def find_k_neighbor_all_nodes(self, node_coords, robot_belief):
        X = node_coords
        if len(node_coords) >= self.k_size:
            knn = NearestNeighbors(n_neighbors=self.k_size)
        else:
            knn = NearestNeighbors(n_neighbors=len(node_coords))
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        for i, p in enumerate(X):
            # for j, neighbour in enumerate(X):
            for j, neighbour in enumerate(X[indices[i][:]]):
                start = p
                end = neighbour
                if (not self.check_collision(start, end, robot_belief)) or (not self.check_collision(end, start, robot_belief)):
                    a = str(self.find_index_from_coords(node_coords, p))
                    b = str(self.find_index_from_coords(node_coords, neighbour))
                    # dist = np.linalg.norm(p - neighbour)
                    # if (int(a) == 41 and int(b) == 35):
                    #     print('dis', dist)
                    if distances[i, j] < self.dist_threshold:
                        self.graph.add_node(a)
                        self.graph.add_edge(a, b, distances[i, j])

                        if self.plot:
                            self.x.append([p[0], neighbour[0]])
                            self.y.append([p[1], neighbour[1]])

    def find_index_from_coords(self, node_coords, p):
        return np.where(np.linalg.norm(node_coords - p, axis=1) < 1e-5)[0][0]

    def find_index_from_coords2(self, node_coords, p):
        index = np.argmin(np.linalg.norm(node_coords - p, axis=1))
        return index

    def check_collision(self, start, end, robot_belief):
        # Bresenham line algorithm checking
        collision = False

        x0 = start[0].round()
        y0 = start[1].round()
        x1 = end[0].round()
        y1 = end[1].round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        while 0 <= x < robot_belief.shape[1] and 0 <= y < robot_belief.shape[0]:
            k = robot_belief.item(int(y), int(x))
            if x == x1 and y == y1:
                break
            if k == 1:
                collision = True
                break
            if k == 127:
                collision = True
                break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return collision

    def find_shortest_path(self, current, destination, node_coords):
        start_node = str(self.find_index_from_coords2(node_coords, current))
        end_node = str(self.find_index_from_coords2(node_coords, destination))
        if start_node == end_node:
            return 0, [start_node]
        route, dist = a_star(int(start_node), int(end_node), self.node_coords, self.graph)
        if route == None:
            return 0, [-1]
        if start_node != end_node:
            assert route != []
        route = list(map(str, route))
        return dist, route
