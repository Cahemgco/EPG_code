import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy
import networkx as nx

from graph import Graph, a_star
from parameter import *
from copy import deepcopy
import os
import random

if not train_mode:
    from test_parameter import *

class Graph_generator:
    def __init__(self, map_size, k_size, plot=False):
        self.k_size = k_size
        self.graph = Graph()
        self.node_coords = None
        self.plot = plot
        self.max_dist = 0
        self.x = []
        self.y = []
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        # self.dist_threshold = (((self.map_x-1)/(NUMX-1))**2 + ((self.map_y-1)/(NUMY-1))**2)**0.5 * 1.1
        # self.uniform_points = self.generate_uniform_points()
        self.nodes_list = []
        self.reference_policy = None
        self.adjacent_matrix = []
        self.network_adjacent_matrix = []

    def generate_graph(self, robot_belief, map_index):
        # get node_coords by finding the uniform points in free area
        if map_index < 10:
            self.numx = 16
            self.numy = 16
        else:
            self.numx = 32
            self.numy = 32

        # self.numx = NUMX
        # self.numy = NUMY
        self.dist_threshold = (((np.shape(robot_belief)[1]-1)/(self.numx-1))**2 + ((np.shape(robot_belief)[0]-1)/(self.numy-1))**2)**0.5 * 1.1
        self.uniform_points = self.generate_uniform_points()
        
        free_area = self.free_area(robot_belief)
        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        node_coords = self.uniform_points[candidate_indices]
        
        self.node_coords = self.unique_coords(node_coords).reshape(-1, 2)
        node_num = len(self.node_coords)

        # generate the collision free graph
        self.find_k_neighbor_all_nodes(self.node_coords, robot_belief)
        
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
        for i in range(len(self.adjacent_matrix)):
            for j in range(len(self.adjacent_matrix)):
                if i == j:
                    dist[i][j] = 0
                elif self.adjacent_matrix[i][j] == 0:
                    dist[i][j] = 1

        for k in range(len(self.adjacent_matrix)):
            for i in range(len(self.adjacent_matrix)):
                for j in range(len(self.adjacent_matrix)):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        next_node = [[j for j in range(len(self.adjacent_matrix))] for i in range(len(self.adjacent_matrix))]
        for i in range(len(self.adjacent_matrix)):
            for j in range(len(self.adjacent_matrix)):
                for k in range(len(self.adjacent_matrix)):
                    if dist[i][k] == 1 and dist[i][j] == dist[k][j] + 1:
                        next_node[i][j] = k

        self.max_dist = np.max(dist)
        self.network_adjacent_matrix = np.array(dist)

        # adj = 1 - self.adjacent_matrix
        # graph_adj = nx.from_numpy_array(adj)
        # print('edge num: ', graph_adj.number_of_edges(), file_path)

        # adjacent_matrix_file = str('adj_file_train/adjacent_matrix_'+file_path+'.txt')
        # with open(adjacent_matrix_file, 'w') as f:
        #     f.write(str(len(self.adjacent_matrix))+'\n')
        #     for i in range(len(self.adjacent_matrix)):
        #         for j in range(len(self.adjacent_matrix)):
        #             if j == len(self.adjacent_matrix)-1:
        #                 f.write(str(int(1-self.adjacent_matrix[i][j])) +'\n')
        #             else:
        #                 f.write(str(int(1-self.adjacent_matrix[i][j])) +' ')

        # calculate the feature as the number of observable frontiers of each node
        # save the observable frontiers to be reused
        # random.seed(map_index+1)
        robot_indexes = random.sample(range(node_num), N_ROBOTS)
        robot_positions = [self.node_coords[robot] for robot in robot_indexes]
        candidates = [index for index in range(node_num) if index not in robot_indexes]
        target_index = random.choice(candidates)
        target_position = self.node_coords[target_index]
        # print('robot_indexes', robot_indexes, 'target_index', target_index)
        
        node_feature = []
        for index in range(len(self.node_coords)):
            feature = []
            feature.append(self.network_adjacent_matrix[index][target_index])
            for robot in range(len(robot_indexes)):
                feature.append(self.network_adjacent_matrix[index][robot_indexes[robot]])
            # print(index, feature)
            node_feature.append(feature)
        self.node_feature = np.array(node_feature)
        
        # print('node_feature', self.node_feature)
        return robot_positions, robot_indexes, target_position, target_index, self.node_coords, self.graph.edges, self.node_feature, robot_indexes, target_index, self.adjacent_matrix, self.network_adjacent_matrix, next_node

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
            node_feature.append(feature)
        self.node_feature = np.array(node_feature)

        return self.node_feature

    def generate_uniform_points(self):
        # x = np.linspace(0, self.map_x - 1, NUMX).round().astype(int) # 55
        # y = np.linspace(0, self.map_y - 1, NUMY).round().astype(int)
        x = np.linspace(0, self.map_x - 1, self.numx).round().astype(int)
        y = np.linspace(0, self.map_y - 1, self.numy).round().astype(int)

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
