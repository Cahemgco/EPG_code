import imageio
import csv
import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from env import Env
from model import PolicyNet
from test_parameter import *
import json
import networkx as nx


class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, test, device='cuda', greedy=False, save_image=False, random_seed=None):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image, test=test, num_robots=N_ROBOTS, random_seed=random_seed, input_type=INPUT_TYPE)
        self.local_policy_net = policy_net
        self.travel_dist = 0
        self.perf_metrics = dict()
        if self.env.input_type == 'map':
            self.robot_position = self.env.start_robot_positions
            self.robot_positions = [self.env.start_robot_positions[i] for i in range(N_ROBOTS)]

    def run_episode_shortest(self, ml, curr_episode):
        # print('curr_episode: ', curr_episode)
        if self.env.input_type == 'map':
            done = False
            u_list = []
            gif_file = []
            
            for exit_index in self.env.exit_indexes:
                if self.save_image:
                    path = gifs_path + '_shortest'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    self.env.plot_env(self.global_step, path, 0)
                
                for i in range(1, ml+1):
                    for robot_index in range(N_ROBOTS):
                        observations = self.get_observations(robot_index)
                        next_position, _ = self.select_node(observations)
                        self.env.step(next_position, robot_index)

                    next_target_index = self.env.next_node[self.env.target_index][exit_index]
                    # reward, done, escaped, self.robot_positions = self.env.step_heuristic(next_robot_indexes, next_target_index)
                    self.env.step(next_target_index, N_ROBOTS+1)
                    done, escaped = self.env.check_done(self.env.robot_indexes)

                    # save a frame
                    if self.save_image:
                        path = gifs_path + '_shortest'
                        if not os.path.exists(path):
                            os.makedirs(path)
                        self.env.plot_env(self.global_step, path, i)
                                    
                    if done or escaped:
                        break
                reward = 0

                if escaped:
                    reward = -1
                if done:
                    reward = 1
                        
                # print('done: ', done, 'i: ', i)
                if escaped == False and i == ml:
                    reward = 1

                u = GAMMA ** i * reward
                u_list.append(u)

                if reward == 1:
                    result = 'success'
                else:
                    result = 'fail'

                # save gif
                if self.save_image:
                    path = gifs_path + '_shortest'
                    # self.make_gif(path, curr_episode)
                    title = '{}/{}_exit_{}_step_{:.4g}_utility_{:.2f}_result_{}.gif'.format(path, curr_episode, exit_index, self.env.stepi, u, result)
                    with imageio.get_writer(title, mode='I', fps=2) as writer:
                        for frame in self.env.frame_files:
                            image = imageio.imread(frame)
                            writer.append_data(image)
                    
                    gif_file.append(title)
                    print('gif complete\n')

                    # Remove files
                    for filename in self.env.frame_files:
                        os.remove(filename)

                self.env.resetenv()

            # print('u: ', u_list, 'worst: ', np.min(u_list))
            self.perf_metrics['evader_shortest_worst_u'] = np.min(u_list)
            self.perf_metrics['evader_shortest_avg_u'] = np.average(u_list)

        else:
            done = False
            u_list = []
            gif_file = []
            
            for exit_index in self.env.exit_indexes:
                for i in range(1, ml+1):
                    
                    for robot_index in range(N_ROBOTS):
                        observations = self.get_observations(robot_index)
                        next_position, _ = self.select_node(observations)
                        self.env.step(next_position, robot_index)

                    next_target_index = self.env.next_node[self.env.target_index][exit_index]
                    self.env.step(next_target_index, N_ROBOTS+1)
                    done, escaped = self.env.check_done(self.env.robot_indexes)

                    if done or escaped:
                        break
                reward = 0

                if escaped:
                    reward = -1
                if done:
                    reward = 1
                        
                # print('done: ', done, 'i: ', i)
                if escaped == False and i == ml:
                    reward = 1

                u = GAMMA ** i * reward
                u_list.append(u)

                if reward == 1:
                    result = 'success'
                else:
                    result = 'fail'

                self.env.resetenv()

            # print('u: ', u_list, 'worst: ', np.min(u_list))
            self.perf_metrics['evader_shortest_worst_u'] = np.min(u_list)
            self.perf_metrics['evader_shortest_avg_u'] = np.average(u_list)

    def run_episode_heuristic(self, ml, curr_episode):
        if self.env.input_type == 'map':
            # print('curr_episode: ', curr_episode)
            done = False
            
            if self.save_image:
                path = gifs_path + '_heuristic'
                if not os.path.exists(path):
                    os.makedirs(path)
                self.env.plot_env(self.global_step, path, 0)
            
            for i in range(1, ml+1):
                observation = self.get_observation_heuristic()

                next_robot_indexes, next_target_index = self.select_node_heuristic(observation)
                for robot_index in range(N_ROBOTS):
                    observations = self.get_observations(robot_index)
                    next_position, _ = self.select_node(observations)
                    self.env.step(next_position, robot_index)

                # target move
                self.env.step(next_target_index, N_ROBOTS+1)
                done, escaped = self.env.check_done(self.env.robot_indexes)

                # save a frame
                if self.save_image:
                    path = gifs_path + '_heuristic'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    self.env.plot_env(self.global_step, path, i)
                            
                if done or escaped:
                    break
            reward = 0

            if escaped:
                reward = -1
            if done:
                reward = 1

            # print('done: ', done, 'i: ', i)
            if escaped == False and i == ml:
                reward = 1
            # print('done: ', done, 'escaped: ', escaped, 'step: ', i, 'reward: ', reward)
            u = GAMMA ** i * reward

            if reward == 1:
                result = 'success'
            else:
                result = 'fail'

            # save gif
            if self.save_image:
                path = gifs_path + '_heuristic'
                # self.make_gif(path, curr_episode)
                title = '{}/{}_step_{:.4g}_utility_{:.2f}_result_{}.gif'.format(path, curr_episode, self.env.stepi, u, result)
                with imageio.get_writer(title, mode='I', fps=2) as writer:
                    for frame in self.env.frame_files:
                        image = imageio.imread(frame)
                        writer.append_data(image)
                
                print('gif complete\n')

                # Remove files
                for filename in self.env.frame_files:
                    os.remove(filename)

                self.env.resetenv()

            # print('u: ', u_list, 'worst: ', np.min(u_list))
            self.perf_metrics['heuristic_u'] = u

        else:
            done = False

            for i in range(1, ml+1):
                observation = self.get_observation_heuristic()

                next_robot_indexes, next_target_index = self.select_node_heuristic(observation)
                for robot_index in range(N_ROBOTS):
                    observations = self.get_observations(robot_index)
                    next_position, _ = self.select_node(observations)
                    self.env.step(next_position, robot_index)

                # target move
                self.env.step(next_target_index, N_ROBOTS+1)
                done, escaped = self.env.check_done(self.env.robot_indexes)
                # print('exit index: ', self.env.exit_indexes, 'robot indexes: ', self.env.robot_indexes, 'target index: ', self.env.target_index)
                if done or escaped:
                    break
            reward = 0

            if escaped:
                reward = -1
            if done:
                reward = 1

            # print('done: ', done, 'i: ', i)
            if escaped == False and i == ml:
                reward = 1
            # print('done: ', done, 'escaped: ', escaped, 'step: ', i, 'reward: ', reward)
            u = GAMMA ** i * reward

            if reward == 1:
                result = 'success'
            else:
                result = 'fail'

            self.perf_metrics['heuristic_u'] = u

    def get_observations(self, robot_index):
        if self.env.input_type == 'map':
            node_coords = copy.deepcopy(self.env.node_coords)
            graph = copy.deepcopy(self.env.graph)
            node_feature = copy.deepcopy(self.env.node_feature)

            # normalize observations
            node_feature = node_feature / self.env.graph_generator.max_dist
            n_nodes = node_coords.shape[0]

            node_feature_inputs = node_feature.reshape((n_nodes, N_ROBOTS + EXIT_NUM + 1))
            node_inputs = node_feature_inputs
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

            # get the node index of the current robot position
            current_node_index = self.env.find_index_from_coords(self.robot_positions[robot_index])
            current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)

            # calculate a mask for padded node
            node_padding_mask = None

            # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
            graph = list(graph.values())
            edge_inputs = []
            for node in graph:
                node_edges = list(map(int, node))
                edge_inputs.append(node_edges)

            adjacent_matrix = self.env.adjacent_matrix

            # edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
            edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

            edge = edge_inputs[current_index]
            while len(edge) < self.k_size:
                edge.append(-1)

            edge_input = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)

            # calculate a mask for the padded edges (denoted by -1)
            edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
            one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
            edge_padding_mask = torch.where(edge_input == -1, one, edge_padding_mask)

            edge_inputs = torch.where(edge_input == -1, 0, edge_input)

            observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        
        else:
            # get observations
            graph = copy.deepcopy(self.env.graph)
            node_feature = copy.deepcopy(self.env.node_feature)
            
            # normalize observations
            node_feature = node_feature / np.max(self.env.network_adjacent_matrix)
            
            # transfer to node inputs tensor
            node_feature_inputs = node_feature.reshape((self.env.node_num, N_ROBOTS + EXIT_NUM + 1))
            
            node_inputs = node_feature_inputs
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device) 

            # get the node index of the current robot position
            current_node_index = self.env.robot_indexes[robot_index]
            current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)

            # calculate a mask for padded node
            node_padding_mask = None

            # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
            edge_inputs = []
            for node in graph.nodes():
                neighbors = list(graph.neighbors(node))
                edge_inputs.append(neighbors)

            adjacent_matrix = self.env.adjacent_matrix
            # print('adj matrix: ', adjacent_matrix)
            edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

            edge = edge_inputs[current_index]
            while len(edge) < self.k_size:
                edge.append(-1)

            edge_input = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)

            # calculate a mask for the padded edges (denoted by -1)
            edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
            one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
            edge_padding_mask = torch.where(edge_input == -1, one, edge_padding_mask)

            edge_inputs = torch.where(edge_input == -1, 0, edge_input)
            observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        
        return observations
    
    def get_all_observations(self):
        # get observations
        node_coords = copy.deepcopy(self.env.node_coords)
        graph = copy.deepcopy(self.env.graph)
        node_feature = copy.deepcopy(self.env.node_feature)
        
        # normalize observations
        node_feature = node_feature / self.env.graph_generator.max_dist

        # transfer to node inputs tensor
        n_nodes = node_coords.shape[0]
        node_feature_inputs = node_feature.reshape((n_nodes, N_ROBOTS + EXIT_NUM + 1))
        # permutation = [0] + [(i + robot_index) % N_ROBOTS + 1 for i in range(N_ROBOTS)]
        node_inputs = node_feature_inputs
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 4)

        # get the node index of the current robot position
        all_current_indexes = []
        for i in range(N_ROBOTS):
            all_current_node_index = self.env.find_index_from_coords(self.robot_positions[i])
            all_current_indexes.append(all_current_node_index)
        all_current_index = torch.tensor(all_current_indexes).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1, N_ROBOTS)
        
        # calculate a mask for padded node
        node_padding_mask = None

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        adjacent_matrix = self.env.adjacent_matrix

        # edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
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

        return all_observations

    def select_node(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask= observations
        with torch.no_grad():
            logp = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask,
                                              edge_padding_mask, edge_mask, self.greedy)
        
        if self.greedy:
            action_index = torch.argmax(logp, dim=1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
        
        next_node_index = edge_inputs[0, 0, action_index.item()]
        if self.env.input_type == 'map':
            next_position = self.env.node_coords[next_node_index]
        else:
            next_position = next_node_index
        # print('log shape: ', logp.shape, 'action: ', action_index)
        assert self.env.adjacent_matrix[next_node_index][current_index] == 0, print('current: ', current_index, 'edge_inputs: ', edge_inputs, 'next: ', next_node_index)
        return next_position, action_index

    def get_observation_heuristic(self):
        target_exit_dists = []
        for index, exit in enumerate(self.env.exit_indexes):
            escape_dist = self.env.network_adjacent_matrix[self.env.target_index][exit]
            target_exit_dists.append(escape_dist)

        sorted_exit_indexes = [x for _, x in sorted(zip(target_exit_dists, self.env.exit_indexes))]
        target_exit_dists = sorted(target_exit_dists)
        robot_exit_dists = np.zeros((N_ROBOTS, EXIT_NUM))
        for i, robot in enumerate(self.env.robot_indexes):
            for j, exit in enumerate(sorted_exit_indexes):
                escape_dist = self.env.network_adjacent_matrix[robot][exit]
                robot_exit_dists[i][j] = escape_dist

        edge = np.zeros((N_ROBOTS, EXIT_NUM))
        for i in range(N_ROBOTS):
            for j in range(EXIT_NUM):
                if robot_exit_dists[i][j] <= target_exit_dists[j]:
                    edge[i][j] = 1
        
        observation = edge, sorted_exit_indexes
        return observation

    def select_node_heuristic(self, observation):
        edge, sorted_exit_indexes = observation
        zero_colum = np.where(np.all(edge == 0, axis=0))[0]
        one_colum = np.where(np.any(edge != 0, axis=0))[0]

        if zero_colum.size > 0:
            # print('zero_colum: ', zero_colum[0])
            target_exit = sorted_exit_indexes[zero_colum[0]]
            
        else:
            no_occupied_exit = []
            for exit in sorted_exit_indexes:
                if exit not in self.env.robot_indexes:
                    no_occupied_exit.append(exit)

            target_exit = no_occupied_exit[0]
            
        next_target_index = self.env.next_node[self.env.target_index][target_exit]

        next_robot_indexes = np.zeros(N_ROBOTS)
        zero_row = np.where(np.all(edge == 0, axis=1))[0]
        one_row = np.where(np.any(edge != 0, axis=1))[0]
        # print(zero_row, one_row)
        
        if zero_row.size > 0:
            for row in zero_row:
                # print('row: ', row)
                next_robot_indexes[row] = self.env.next_node[self.env.robot_indexes[row]][self.env.target_index]
                
        robot_nodes = []
        for idx, i in enumerate(one_row):
            node = self.env.robot_indexes[i]
            robot_nodes.append('p'+ str(i) + str(node))
        # print('robot nodes: ', robot_nodes)

        visited_exits = []
        chosen_exit = np.zeros(N_ROBOTS)
        matched_robots_node = []
        for exit_idx, exit_node in enumerate(sorted_exit_indexes):
            temp_matched_robots_node = []
            graph = nx.Graph()
            visited_exits.append('e' + str(exit_idx) + str(exit_node))

            graph.add_nodes_from(robot_nodes, bipartite=0)
            graph.add_nodes_from(visited_exits, bipartite=1)

            for i, robot in enumerate(robot_nodes):
                for j, exit in enumerate(visited_exits):
                    robot_index = self.env.robot_indexes.index(int(robot[2:]))
                    exit_index = sorted_exit_indexes.index(int(exit[2:]))
                    if edge[robot_index, exit_index] == 1:
                        graph.add_edge(robot_nodes[i], visited_exits[j])

            matching = nx.algorithms.bipartite.matching.maximum_matching(graph, top_nodes=robot_nodes)
            matching_pairs = [(u, v) for u, v in matching.items() if u in robot_nodes]

            if len(matching_pairs) < exit_idx + 1:
                break
            
            for idx, pair in enumerate(matching_pairs):
                robot_index_in_map = int(pair[0][2:])
                robot_index = int(pair[0][1])
                exit_index = int(pair[1][2:])

                temp_matched_robots_node.append(pair[0])
                matched_robots_node = copy.deepcopy(temp_matched_robots_node)
                next_robot_indexes[robot_index] = self.env.next_node[robot_index_in_map][exit_index]

        unmatched_robots_node = [robot_node for robot_node in robot_nodes if robot_node not in matched_robots_node]

        for idx, robot_node in enumerate(unmatched_robots_node):  
            robot_index = int(robot_node[1])
            robot_index_in_map = int(robot_node[2:])
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

    def calculate_edge_mask(self, edge_inputs):
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
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
        with imageio.get_writer('{}/{}_step_{:.4g}.gif'.format(path, n, self.perf_metrics['steps']), mode='I', fps=2) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)

        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files:
            os.remove(filename)


    def work(self, curr_episode):
        self.run_episode_shortest(ml=10, curr_episode=curr_episode)
        # self.env.resetenv()
        # self.run_episode_heuristic(ml=10, curr_episode=curr_episode)