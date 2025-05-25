import copy
import os
import networkx as nx

import imageio
import numpy as np
import torch
from env import Env
from parameter import *


class Worker:
    def __init__(self, meta_agent_id, policy_net, q_net, global_step, device='cuda', greedy=False, save_image=False, random_seed=None):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image, num_robots=N_ROBOTS, random_seed=random_seed)
        self.local_policy_net = policy_net
        self.local_q_net = q_net

        self.current_node_index = 0
        self.travel_dist = 0
        if self.env.input_type == 'map':
            self.robot_positions = [self.env.start_robot_positions[i] for i in range(N_ROBOTS)]

        # if FIXED_OPPONENT == False:
        #     self.opponent_position = OPPONENT_POLICIES[self.env.map_index]

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(16):
            self.episode_buffer.append([])

    def get_observations(self, robot_index):
        if self.env.input_type == 'map':
            # get observations
            node_coords = copy.deepcopy(self.env.node_coords)
            graph = copy.deepcopy(self.env.graph)
            node_feature = copy.deepcopy(self.env.node_feature)
            
            # normalize observations
            # 距离
            node_feature = node_feature / self.env.graph_generator.max_dist

            # transfer to node inputs tensor
            n_nodes = node_coords.shape[0]
            
            node_feature_inputs = node_feature.reshape((n_nodes, N_ROBOTS + EXIT_NUM + 1))
            node_inputs = node_feature_inputs
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 4)
            # print('node inputs: ', node_inputs)

            # padding the number of node to a given node padding size
            assert node_coords.shape[0] < self.node_padding_size
            padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
            node_inputs = padding(node_inputs)

            # calculate a mask to padded nodes
            node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
            node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
                self.device)
            node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

            # get the node index of the current robot position
            current_node_index = self.env.find_index_from_coords(self.robot_positions[robot_index])
            current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)

            # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
            graph = list(graph.values())
            edge_inputs = []
            for node in graph:
                node_edges = list(map(int, node))
                edge_inputs.append(node_edges)
            edge_for_select_node = edge_inputs

            adjacent_matrix = self.env.adjacent_matrix

            # edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
            edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

            # padding edge mask
            assert len(edge_inputs) < self.node_padding_size
            padding = torch.nn.ConstantPad2d(
                (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
            edge_mask = padding(edge_mask)

            edge = edge_inputs[current_index]
            while len(edge) < self.k_size:
                edge.append(-1)

            edge_input = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
            edge_inputs = torch.where(edge_input == -1, 0, edge_input)

            # calculate a mask for the padded edges (denoted by -1)
            edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
            one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
            edge_padding_mask = torch.where(edge_input == -1, one, edge_padding_mask)
            # print('edge_padding_mask: ', edge_padding_mask.shape)
            observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        else:
            # get observations
            graph = copy.deepcopy(self.env.graph)
            node_feature = copy.deepcopy(self.env.node_feature)
            
            # normalize observations
            node_feature = node_feature / np.max(self.env.network_adjacent_matrix)
            
            # transfer to node inputs tensor
            node_feature_inputs = node_feature.reshape((self.env.node_num, N_ROBOTS + EXIT_NUM + 1))
            # permutation = [0] + [(i + robot_index) % N_ROBOTS + 1 for i in range(N_ROBOTS)]
            node_inputs = node_feature_inputs
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 4)

            # padding the number of node to a given node padding size
            assert self.env.node_num < self.node_padding_size
            padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - self.env.node_num))
            node_inputs = padding(node_inputs)

            # calculate a mask for padded node
            node_padding_mask = torch.zeros((1, 1, self.env.node_num), dtype=torch.int64).to(self.device)
            node_padding = torch.ones((1, 1, self.node_padding_size - self.env.node_num), dtype=torch.int64).to(
                self.device)
            node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

            # get the node index of the current robot position
            current_node_index = self.env.robot_indexes[robot_index]
            current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)

            # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
            edge_inputs = []
            for node in graph.nodes():
                neighbors = list(graph.neighbors(node))
                edge_inputs.append(neighbors)
            edge_for_select_node = edge_inputs

            adjacent_matrix = self.env.adjacent_matrix
            # print('adj matrix: ', adjacent_matrix)
            edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

            # padding edge mask
            assert len(edge_inputs) < self.node_padding_size
            padding = torch.nn.ConstantPad2d(
                (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
            edge_mask = padding(edge_mask)

            edge = edge_inputs[current_index]
            while len(edge) < self.k_size:
                edge.append(-1)

            edge_input = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
            edge_inputs = torch.where(edge_input == -1, 0, edge_input)

            # calculate a mask for the padded edges (denoted by -1)
            edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
            one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
            edge_padding_mask = torch.where(edge_input == -1, one, edge_padding_mask)
            observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask

        return observations, edge_for_select_node
    
    def get_all_observations(self):
        if self.env.input_type == 'map':
            # get observations
            node_coords = copy.deepcopy(self.env.node_coords)
            graph = copy.deepcopy(self.env.graph)
            node_feature = copy.deepcopy(self.env.node_feature)
            
            # normalize observations
            # 距离
            node_feature = node_feature / self.env.graph_generator.max_dist

            # transfer to node inputs tensor
            n_nodes = node_coords.shape[0]
            
            node_feature_inputs = node_feature.reshape((n_nodes, N_ROBOTS + EXIT_NUM + 1))
            node_inputs = node_feature_inputs
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 4)
            # print('node inputs: ', node_inputs)

            # padding the number of node to a given node padding size
            assert node_coords.shape[0] < self.node_padding_size
            padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
            node_inputs = padding(node_inputs)

            # calculate a mask to padded nodes
            node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
            node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
                self.device)
            node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

            # get the node index of the current robot position
            all_current_indexes = []
            for i in range(N_ROBOTS):
                all_current_node_index = self.env.find_index_from_coords(self.robot_positions[i])
                all_current_indexes.append(all_current_node_index)
            all_current_index = torch.tensor(all_current_indexes).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1, N_ROBOTS)
            
            # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
            graph = list(graph.values())
            edge_inputs = []
            for node in graph:
                node_edges = list(map(int, node))
                edge_inputs.append(node_edges)
                assert node_edges != -1
            edge_for_select_node = edge_inputs

            adjacent_matrix = self.env.adjacent_matrix

            # edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
            edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

            # padding edge mask
            assert len(edge_inputs) < self.node_padding_size
            padding = torch.nn.ConstantPad2d(
                (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
            edge_mask = padding(edge_mask)

            all_edges_inputs = []
            all_edge_padding_masks = []
            # print('all_current_indexes: ', all_current_indexes)
            for i in range(N_ROBOTS):
                edge = edge_inputs[all_current_indexes[i]]
                while len(edge) < self.k_size:
                    edge.append(-1)

                edge_input = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
                
                # calculate a mask for the padded edges (denoted by -1)
                edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
                one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
                edge_padding_mask = torch.where(edge_input == -1, one, edge_padding_mask)
                edge_input = torch.where(edge_input == -1, 0, edge_input)
                all_edges_inputs.append(edge_input)
                all_edge_padding_masks.append(edge_padding_mask)

            all_edge_inputs = torch.cat(all_edges_inputs, dim=-1)
            all_edge_padding_mask = torch.cat(all_edge_padding_masks, dim=-1)
            all_observations = node_inputs, all_edge_inputs, all_current_index, node_padding_mask, all_edge_padding_mask, edge_mask
        else:
            # get observations
            graph = copy.deepcopy(self.env.graph)
            node_feature = copy.deepcopy(self.env.node_feature)
            
            # normalize observations
            node_feature = node_feature / np.max(self.env.network_adjacent_matrix)
            
            # transfer to node inputs tensor
            node_feature_inputs = node_feature.reshape((self.env.node_num, N_ROBOTS + EXIT_NUM + 1))
            # permutation = [0] + [(i + robot_index) % N_ROBOTS + 1 for i in range(N_ROBOTS)]
            node_inputs = node_feature_inputs
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 4)

            # padding the number of node to a given node padding size
            assert self.env.node_num < self.node_padding_size
            padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - self.env.node_num))
            node_inputs = padding(node_inputs)

            # calculate a mask for padded node
            node_padding_mask = torch.zeros((1, 1, self.env.node_num), dtype=torch.int64).to(self.device)
            node_padding = torch.ones((1, 1, self.node_padding_size - self.env.node_num), dtype=torch.int64).to(
                self.device)
            node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

            # get the node index of the current robot position
            all_current_indexes = []
            for i in range(N_ROBOTS):
                all_current_node_index = self.env.robot_indexes[i]
                all_current_indexes.append(all_current_node_index)
            all_current_index = torch.tensor(all_current_indexes).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1, N_ROBOTS)

            # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
            edge_inputs = []
            for node in graph.nodes():
                neighbors = list(graph.neighbors(node))
                edge_inputs.append(neighbors)
            edge_for_select_node = edge_inputs

            adjacent_matrix = self.env.adjacent_matrix
            # print('adj matrix: ', adjacent_matrix)
            edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

            # padding edge mask
            assert len(edge_inputs) < self.node_padding_size
            padding = torch.nn.ConstantPad2d(
                (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
            edge_mask = padding(edge_mask)

            all_edges_inputs = []
            all_edge_padding_masks = []
            for i in range(N_ROBOTS):
                edge = edge_inputs[all_current_indexes[i]]
                while len(edge) < self.k_size:
                    edge.append(-1)

                edge_input = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
                
                # calculate a mask for the padded edges (denoted by -1)
                edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
                one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
                edge_padding_mask = torch.where(edge_input == -1, one, edge_padding_mask)
                edge_input = torch.where(edge_input == -1, 0, edge_input)
                all_edges_inputs.append(edge_input)
                all_edge_padding_masks.append(edge_padding_mask)

            all_edge_inputs = torch.cat(all_edges_inputs, dim=-1)
            all_edge_padding_mask = torch.cat(all_edge_padding_masks, dim=-1)
            all_observations = node_inputs, all_edge_inputs, all_current_index, node_padding_mask, all_edge_padding_mask, edge_mask

        return all_observations, edge_for_select_node
    
    def get_observation_heuristic(self):
        # 逃逸者到出口的距离
        target_exit_dists = []
        for index, exit in enumerate(self.env.exit_indexes):
            # escape_dist_1 = nx.shortest_path_length(G=graph, source=self.env.target_index, target=exit)
            escape_dist = self.env.network_adjacent_matrix[self.env.target_index][exit]
            # assert escape_dist == escape_dist_1
            target_exit_dists.append(escape_dist)
        # print('target_exit_dists: ', target_exit_dists)

        # 按照逃逸者到出口的距离，对出口从小到大进行排序
        sorted_exit_indexes = [x for _, x in sorted(zip(target_exit_dists, self.env.exit_indexes))]
        target_exit_dists = sorted(target_exit_dists)
        # print('target_exit_dists: ', target_exit_dists)
        # print('exit: ', self.env.exit_indexes, 'sorted_exit_indexes: ', sorted_exit_indexes)

        # 追捕者到出口的距离
        robot_exit_dists = np.zeros((N_ROBOTS, EXIT_NUM))
        for i, robot in enumerate(self.env.robot_indexes):
            for j, exit in enumerate(sorted_exit_indexes):
                # num = self.env.exit_indexes.index(exit)
                escape_dist = self.env.network_adjacent_matrix[robot][exit]
                # assert escape_dist == escape_dist_2
                robot_exit_dists[i][j] = escape_dist

        # 计算 edge 矩阵并添加边
        edge = np.zeros((N_ROBOTS, EXIT_NUM))  # 初始化边矩阵
        for i in range(N_ROBOTS):
            for j in range(EXIT_NUM):
                if robot_exit_dists[i][j] <= target_exit_dists[j]:
                    edge[i][j] = 1  # 标记为可匹配
                    # G.add_edge(robot_nodes[i], exit_nodes[j])  # 同时添加到图中
        # print('edge: \n', edge)
        
        observation = edge, sorted_exit_indexes
        return observation

    def select_node(self, robot_index, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
    
        with torch.no_grad():
            logp = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask,
                                                edge_padding_mask, edge_mask)

        if self.greedy:
            action_index = torch.argmax(logp, dim=1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
        next_node_index = edge_inputs[0, 0, action_index.item()]
        if self.env.input_type == 'map':
            next_position = self.env.node_coords[next_node_index]
        else:
            next_position = next_node_index
        assert self.env.adjacent_matrix[next_node_index][current_index] == 0, print('current: ', current_index, 'edge_inputs: ', edge_inputs, 'next: ', next_node_index, 'robot: ', robot_index)
        return next_position, action_index

    def select_node_random(self, robot, edge_for_select_node):
        # node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        robot_index = self.env.robot_indexes[robot]
        # assert -1 not in edge_for_select_node[robot_index], print('edge_for_select_node: ', edge_for_select_node[robot_index])
        candidate = [edge_for_select_node[robot_index][i] for i in range(len(edge_for_select_node[robot_index])) if edge_for_select_node[robot_index][i] != -1]
        next_node_index = np.random.choice(candidate)
        # print('next_node_index: ', next_node_index)

        action_index = edge_for_select_node[robot_index].index(next_node_index)
        # print('action_index: ', action_index)

        if self.env.input_type == 'map':
            next_position = self.env.node_coords[next_node_index]
        else:
            next_position = next_node_index
        assert self.env.adjacent_matrix[next_node_index][robot_index] == 0, print('current: ', robot_index, 'edge_inputs: ', edge_for_select_node, 'next: ', next_node_index, 'robot: ', robot)
        # print('next_position: ', next_position, 'action_index: ', action_index)
        return next_position, action_index
    
    def select_node_heuristic(self, observation):
        edge, sorted_exit_indexes = observation
        # 逃逸者策略
        zero_colum = np.where(np.all(edge == 0, axis=0))[0]    # 判断是否有出口无连边
        one_colum = np.where(np.any(edge != 0, axis=0))[0]
        # print('zero_colum: ', zero_colum, 'one_colum: ', one_colum)
        if zero_colum.size > 0:
            # print('zero_colum: ', zero_colum[0])
            target_exit = sorted_exit_indexes[zero_colum[0]]
            
        else:
            # 随机选取一个有边的出口
            # candidate = [sorted_exit_indexes[colum] for colum in one_colum]
            # target_exit = np.random.choice(candidate)

            # 选取最近的未被占据的出口
            no_occupied_exit = []
            for exit in sorted_exit_indexes:
                if exit not in self.env.robot_indexes:
                    no_occupied_exit.append(exit)

            target_exit = no_occupied_exit[0]
            
        next_target_index = self.env.next_node[self.env.target_index][target_exit]
        # print('path length: ', len(evade_path))
        # print('target_exit: ', target_exit, 'next_target_index: ', next_target_index)

        # 追捕者策略
        next_robot_indexes = np.zeros(N_ROBOTS)
        zero_row = np.where(np.all(edge == 0, axis=1))[0]    # 判断是否有追捕者无连边
        one_row = np.where(np.any(edge != 0, axis=1))[0]    # 判断是否有追捕者有连边
        # print(zero_row, one_row)
        
        if zero_row.size > 0:
            for row in zero_row:
                # print('row: ', row)
                next_robot_indexes[row] = self.env.next_node[self.env.robot_indexes[row]][self.env.target_index]

        # 优先级匹配策略
        robot_nodes = []
        for idx, i in enumerate(one_row):
            node = self.env.robot_indexes[i]
            robot_nodes.append('p'+ str(i) + str(node))
        # print('robot nodes: ', robot_nodes)

        visited_exits = []
        chosen_exit = np.zeros(N_ROBOTS)
        matched_robots_node = []
        for exit_idx, exit_node in enumerate(sorted_exit_indexes):
            # temp_matched_robots = []
            temp_matched_robots_node = []
            graph = nx.Graph()
            visited_exits.append('e' + str(exit_idx) + str(exit_node))
            # print('visited exits: ', visited_exits)

            # 创建目前的二分图
            graph.add_nodes_from(robot_nodes, bipartite=0)  # 左侧：机器人
            graph.add_nodes_from(visited_exits, bipartite=1)  # 右侧：出口

            # 计算 edge 矩阵并添加边
            for i, robot in enumerate(robot_nodes):
                for j, exit in enumerate(visited_exits):
                    robot_index = self.env.robot_indexes.index(int(robot[2:]))
                    exit_index = sorted_exit_indexes.index(int(exit[2:]))
                    if edge[robot_index, exit_index] == 1:
                        graph.add_edge(robot_nodes[i], visited_exits[j])

            # 求最大匹配
            matching = nx.algorithms.bipartite.matching.maximum_matching(graph, top_nodes=robot_nodes)
            matching_pairs = [(u, v) for u, v in matching.items() if u in robot_nodes]
            # print("最大匹配数:", len(matching_pairs))
            # print("匹配对:", matching_pairs)

            if len(matching_pairs) < exit_idx + 1:
                break
            
            for idx, pair in enumerate(matching_pairs):
                robot_index_in_map = int(pair[0][2:])
                robot_index = int(pair[0][1])
                exit_index = int(pair[1][2:])
                # print(robot_index_in_map, exit_index)

                # temp_matched_robots.append(robot_index_in_map)
                temp_matched_robots_node.append(pair[0])
                # matched_robots = copy.deepcopy(temp_matched_robots)
                matched_robots_node = copy.deepcopy(temp_matched_robots_node)
                next_robot_indexes[robot_index] = self.env.next_node[robot_index_in_map][exit_index]

        # print('所有匹配上的追捕者: ', matched_robots_node)

        # 未匹配追捕者策略
        unmatched_robots_node = [robot_node for robot_node in robot_nodes if robot_node not in matched_robots_node]
        # print('未匹配的追捕者: ', unmatched_robots_node)

        # 遍历 unmatched_robots
        for idx, robot_node in enumerate(unmatched_robots_node):  
            robot_index = int(robot_node[1])
            robot_index_in_map = int(robot_node[2:])
            # 筛选与该追捕者相连的出口
            connected_exits = [exit for j, exit in enumerate(sorted_exit_indexes) if edge[robot_index][j] == 1]
            # print('connected_exits: ', connected_exits)
            if connected_exits:
                # distances_to_exits = [
                #     self.env.network_adjacent_matrix[robot_index_in_map][exit]
                #     for exit in connected_exits
                # ]
                # closest_exit = connected_exits[np.argmin(distances_to_exits)]
                closest_exit = connected_exits[0]
            else:
                closest_exit = self.env.target_index

            chosen_exit[robot_index] = closest_exit
            next_robot_indexes[robot_index] = self.env.next_node[robot_index_in_map][closest_exit]

        next_robot_indexes = next_robot_indexes.astype(int)

        # assert 0 not in next_robot_indexes
        return next_robot_indexes, next_target_index

    def save_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask= observations
        self.episode_buffer[0] += copy.deepcopy(node_inputs)
        self.episode_buffer[1] += copy.deepcopy(edge_inputs)
        self.episode_buffer[2] += copy.deepcopy(current_index)
        self.episode_buffer[3] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[4] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[5] += copy.deepcopy(edge_mask).bool()


    def save_action(self, action_index):
        action_index = torch.tensor(action_index)
        self.episode_buffer[6] += copy.deepcopy(action_index.unsqueeze(0))

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += copy.deepcopy(torch.FloatTensor([[[reward]]]).to(self.device))
        self.episode_buffer[8] += copy.deepcopy(torch.tensor([[[(int(done))]]]).to(self.device))

    def save_next_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        self.episode_buffer[9] += copy.deepcopy(node_inputs)
        self.episode_buffer[10] += copy.deepcopy(edge_inputs)
        self.episode_buffer[11] += copy.deepcopy(current_index)
        self.episode_buffer[12] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[13] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[14] += copy.deepcopy(edge_mask).bool()

    def save_reference_policy(self, reference_policy):
        reference_policy = torch.tensor(reference_policy)
        self.episode_buffer[15] += copy.deepcopy(reference_policy.unsqueeze(0))

    def run_episode(self, curr_episode):
        if self.env.input_type == 'map':
            done = False
            reward = 0
            for i in range(128):
                # 脱逃者的视角
                # previous_index = copy.deepcopy(self.env.robot_indexes)
                # # target move
                # self.env.step(None, N_ROBOTS+1, previous_index)
                initial_all_observations, edge_for_select_node = self.get_all_observations()
                # print('edge_for_select_node: ', edge_for_select_node)
                action_index_list = []
                rp_list = []
                self.save_observations(initial_all_observations)
                
                # 启发式
                heuristic_obs = self.get_observation_heuristic()
                next_robot_indexes, next_target_index = self.select_node_heuristic(heuristic_obs)
                for robot, next_index in enumerate(next_robot_indexes):
                    rp = edge_for_select_node[self.env.robot_indexes[robot]].index(next_index)
                    rp = torch.tensor(rp).unsqueeze(0).unsqueeze(0)
                    rp_list.append(rp)

                for robot_index in range(N_ROBOTS):
                    observations, _ = self.get_observations(robot_index)
                    next_positions, action_index = self.select_node(robot_index, observations)
                    action_index_list.append(action_index)
                    _, _, self.robot_positions = self.env.step(next_positions, robot_index)

                # target move
                self.env.step(next_target_index, N_ROBOTS+1)
                done, escaped = self.env.check_done(self.env.robot_indexes)
                reward = 0
                
                if escaped:
                    reward = -30
                if done:
                    reward = 0

                if escaped:
                    done = True
                self.save_action(action_index_list)
                self.save_reference_policy(rp_list)
                self.save_reward_done(reward, done)

                final_all_observations, _ = self.get_all_observations()
                self.save_next_observations(final_all_observations)

                # save a frame
                if self.save_image:
                    if not os.path.exists(gifs_path):
                        os.makedirs(gifs_path)
                    self.env.plot_env(self.global_step, gifs_path, i)

                if done or escaped:
                    break
            # u = GAMMA ** (i + 1) * reward
            # save metrics
            self.perf_metrics['success_rate'] = done
            self.perf_metrics['total_steps'] = i
            # self.perf_metrics['heuristic_utility'] = u

            # save gif
            if self.save_image:
                path = gifs_path
                self.make_gif(path, curr_episode)
        else:
            done = False
            reward = 0
            for i in range(128):
                # previous_index = copy.deepcopy(self.env.robot_indexes)
                # # target move
                # self.env.step(None, N_ROBOTS+1, previous_index)
                initial_all_observations, edge_for_select_node = self.get_all_observations()
                action_index_list = []
                rp_list = []
                self.save_observations(initial_all_observations)

                # 启发式
                heuristic_obs = self.get_observation_heuristic()
                next_robot_indexes, next_target_index = self.select_node_heuristic(heuristic_obs)
                # print('robot_indexes: ', self.env.robot_indexes, 'target: ', self.env.target_index)
                # print('next_robot_indexes: ', next_robot_indexes, 'next target: ', next_target_index)
                for robot, next_index in enumerate(next_robot_indexes):
                    # print("self.env.robot_indexes[robot]: ", self.env.robot_indexes[robot])
                    rp = edge_for_select_node[self.env.robot_indexes[robot]].index(next_index)
                    rp = torch.tensor(rp).unsqueeze(0).unsqueeze(0)
                    rp_list.append(rp)

                for robot_index in range(N_ROBOTS):
                    observations, _ = self.get_observations(robot_index)
                    next_index, action_index = self.select_node(robot_index, observations)

                    action_index_list.append(action_index)

                    self.env.step(next_index, robot_index)
                # target move
                self.env.step(next_target_index, N_ROBOTS+1)
                done, escaped = self.env.check_done(self.env.robot_indexes)
                reward = 0
                
                if escaped:
                    reward = -30
                if done:
                    reward = 0

                self.save_action(action_index_list)
                self.save_reference_policy(rp_list)
                self.save_reward_done(reward, done)

                final_all_observations, _ = self.get_all_observations()
                self.save_next_observations(final_all_observations)

                # # target move
                # self.env.step(None, N_ROBOTS+1, previous_index)
                
                if done or escaped:
                    break

            self.perf_metrics['success_rate'] = done
            self.perf_metrics['total_steps'] = i

    def generate_batch_data(self):
    
        target_index = np.random.choice(self.env.node_num)
        robot_indexes = np.random.choice(self.env.node_num, N_ROBOTS)
        # print('target: ', target_index, 'robots: ', robot_indexes)
        self.env.target_index = target_index
        self.env.robot_indexes = list(robot_indexes)
        if INPUT_TYPE == 'map':
            self.env.robot_positions = [self.env.node_coords[robot_index] for robot_index in robot_indexes]
            self.env.target_position = self.env.node_coords[target_index]
            self.env.node_feature = self.env.graph_generator.update_graph(robot_indexes, target_index)
        else:
            node_feature = []
            for index in range(self.env.node_num):
                feature = []
                feature.append(self.env.network_adjacent_matrix[index][self.env.target_index])
                for robot in range(self.env.num_robots):
                    feature.append(self.env.network_adjacent_matrix[index][self.env.robot_indexes[robot]])
                if EXIT_NUM > 0:
                    for exit in self.env.exit_indexes:
                        feature.append(self.env.network_adjacent_matrix[index][exit])
                # print(index, feature)
                node_feature.append(feature)
            self.env.node_feature = np.array(node_feature)
                

        initial_all_observations, edge_for_select_node = self.get_all_observations()
        action_index_list = []
        rp_list = []
        self.save_observations(initial_all_observations)

        # 启发式
        heuristic_obs = self.get_observation_heuristic()
        next_robot_indexes, next_target_index = self.select_node_heuristic(heuristic_obs)
        for robot, next_index in enumerate(next_robot_indexes):
            rp = edge_for_select_node[self.env.robot_indexes[robot]].index(next_index)
            rp = torch.tensor(rp).unsqueeze(0).unsqueeze(0)
            rp_list.append(rp)

        for robot_index in range(N_ROBOTS):
            observations, edge_for_select_node = self.get_observations(robot_index)
            next_positions, action_index = self.select_node_random(robot_index, edge_for_select_node)
            action_index_list.append(action_index)
            self.env.step(next_positions, robot_index)
                
        self.env.step(next_target_index, N_ROBOTS+1)
        done, escaped = self.env.check_done(self.env.robot_indexes)
        reward = 0
        
        if escaped:
            reward = -30
        if done:
            reward = 0

        if escaped:
            done = True
        self.save_action(action_index_list)
        self.save_reference_policy(rp_list)
        self.save_reward_done(reward, done)

        final_all_observations, _ = self.get_all_observations()
        self.save_next_observations(final_all_observations)
            
    
    # def run_episode_shortest(self, ml, curr_episode):
    #     # print('curr_episode: ', curr_episode)
    #     done = False
    #     u_list = []
    #     gif_file = []
        
    #     for exit_index in self.env.exit_indexes:
    #         if self.save_image:
    #             path = gifs_path + '_shortest'
    #             if not os.path.exists(path):
    #                 os.makedirs(path)
    #             self.env.plot_env(self.global_step, path, 0)
            
    #         for i in range(1, ml+1):
    #             # observation = self.get_observation_heuristic()
                                
    #             # next_robot_indexes, _ = self.select_node_heuristic(observation)
    #             for robot_index in range(N_ROBOTS):
    #                 observations = self.get_observations(robot_index)
    #                 next_position, _ = self.select_node(observations)
    #                 self.env.step(next_position, robot_index)

    #             next_target_index = self.env.next_node[self.env.target_index][exit_index]
    #             # reward, done, escaped, self.robot_positions = self.env.step_heuristic(next_robot_indexes, next_target_index)
    #             self.env.step(next_target_index, N_ROBOTS+1)
    #             done, escaped = self.env.check_done(self.env.robot_indexes)

    #             # save a frame
    #             if self.save_image:
    #                 path = gifs_path + '_shortest'
    #                 if not os.path.exists(path):
    #                     os.makedirs(path)
    #                 self.env.plot_env(self.global_step, path, i)
                                
    #             if done or escaped:
    #                 break
    #         reward = 0

    #         if escaped:
    #             reward = -1
    #         if done:
    #             reward = 1
                    
    #         # print('done: ', done, 'i: ', i)
    #         if escaped == False and i == ml:
    #             reward = 1

    #         u = GAMMA ** i * reward
    #         u_list.append(u)

    #         if reward == 1:
    #             result = 'success'
    #         else:
    #             result = 'fail'

    #         # save gif
    #         if self.save_image:
    #             path = gifs_path + '_shortest'
    #             # self.make_gif(path, curr_episode)
    #             title = '{}/{}_exit_{}_step_{:.4g}_utility_{:.2f}_result_{}.gif'.format(path, curr_episode, exit_index, self.env.stepi, u, result)
    #             with imageio.get_writer(title, mode='I', fps=2) as writer:
    #                 for frame in self.env.frame_files:
    #                     image = imageio.imread(frame)
    #                     writer.append_data(image)
                
    #             gif_file.append(title)
    #             print('gif complete\n')

    #             # Remove files
    #             for filename in self.env.frame_files:
    #                 os.remove(filename)

    #         self.env.resetenv()

    #     # print('u: ', u_list, 'worst: ', np.min(u_list))
    #     self.perf_metrics['evader_shortest_worst_u'] = np.min(u_list)
    #     self.perf_metrics['evader_shortest_avg_u'] = np.average(u_list)

    # def run_episode_heuristic(self, ml, curr_episode):
    #     # print('curr_episode: ', curr_episode)
    #     done = False
        
    #     if self.save_image:
    #         path = gifs_path + '_heuristic'
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         self.env.plot_env(self.global_step, path, 0)
        
    #     for i in range(1, ml+1):
    #         observation = self.get_observation_heuristic()

    #         next_robot_indexes, next_target_index = self.select_node_heuristic(observation)
    #         for robot_index in range(N_ROBOTS):
    #             next_position = self.env.node_coords[next_robot_indexes[robot_index]]
    #             self.env.step(next_position, robot_index)

    #         # target move
    #         self.env.step(next_target_index, N_ROBOTS+1)
    #         done, escaped = self.env.check_done(self.env.robot_indexes)

    #         # save a frame
    #         if self.save_image:
    #             path = gifs_path + '_heuristic'
    #             if not os.path.exists(path):
    #                 os.makedirs(path)
    #             self.env.plot_env(self.global_step, path, i)
                        
    #         if done or escaped:
    #             break
    #     reward = 0

    #     if escaped:
    #         reward = -1
    #     if done:
    #         reward = 1

    #     # print('done: ', done, 'i: ', i)
    #     if escaped == False and i == ml:
    #         reward = 1
    #     # print('done: ', done, 'escaped: ', escaped, 'step: ', i, 'reward: ', reward)
    #     u = GAMMA ** i * reward

    #     if reward == 1:
    #         result = 'success'
    #     else:
    #         result = 'fail'

    #     # save gif
    #     if self.save_image:
    #         path = gifs_path + '_heuristic'
    #         # self.make_gif(path, curr_episode)
    #         title = '{}/{}_step_{:.4g}_utility_{:.2f}_result_{}.gif'.format(path, curr_episode, self.env.stepi, u, result)
    #         with imageio.get_writer(title, mode='I', fps=2) as writer:
    #             for frame in self.env.frame_files:
    #                 image = imageio.imread(frame)
    #                 writer.append_data(image)
            
    #         print('gif complete\n')

    #         # Remove files
    #         for filename in self.env.frame_files:
    #             os.remove(filename)

    #         self.env.resetenv()

    #     # print('u: ', u_list, 'worst: ', np.min(u_list))
    #     self.perf_metrics['heuristic_u'] = u

    def work(self, currEpisode):
        self.run_episode(currEpisode)

    # def work_test(self, curr_episode):
    #     self.run_episode_shortest(ml=10, curr_episode=curr_episode)
    #     self.env.resetenv()
    #     self.run_episode_heuristic(ml=10, curr_episode=curr_episode)

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
    
    def calculate_A_star_edge_mask(self, edge_inputs, matrix):
        # print('edge_inputs:', edge_inputs)
        size = len(edge_inputs)
        bias_matrix = np.copy(matrix)
        for i in range(size):
            cnt = 0
            cnt2 = 0
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
                if bias_matrix[i][j] == 0:
                    cnt += 1
                if matrix[i][j] == 0:
                    cnt2 += 1
            print('cnt:', cnt, 'cnt2:', cnt2)
            if cnt > K_SIZE:
                print('i:', i, 'edge_inputs:', edge_inputs[i], matrix[i])
            assert cnt <= K_SIZE
        return bias_matrix
    
    def calculate_attention_matrix(self, node_utility):
        size = len(node_utility)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for k in range(N_ROBOTS):
                if node_utility[i, k + 1] == 0:
                    for j in range(size):
                        # node with utility > 0
                        if node_utility[j, 0] > 0:
                            bias_matrix[i][j] = 0
                            bias_matrix[j][i] = 0
                        # node with a robot
                        for l in range(N_ROBOTS):
                            if node_utility[j, l + 1] == 0:
                                bias_matrix[i][j] = 0
                                bias_matrix[j][i] = 0
                                break
                    break
        return bias_matrix

    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_step_{:.4g}.gif'.format(path, n, self.perf_metrics['total_steps']), mode='I', fps=2) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)

        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files:
            os.remove(filename)



