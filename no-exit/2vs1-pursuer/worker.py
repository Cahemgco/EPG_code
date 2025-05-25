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
            self.robot_positions = [self.env.start_position for _ in range(N_ROBOTS)]

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
            node_feature = node_feature / self.env.graph_generator.max_dist

            # transfer to node inputs tensor
            n_nodes = node_coords.shape[0]
            node_feature_inputs = node_feature.reshape((n_nodes, N_ROBOTS + 1))
            node_inputs = node_feature_inputs
            
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 4)

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
        else:
            # get observations
            graph = copy.deepcopy(self.env.graph)
            node_feature = copy.deepcopy(self.env.node_feature)
            
            # normalize observations
            node_feature = node_feature / np.max(self.env.network_adjacent_matrix)
            
            # transfer to node inputs tensor
            node_feature_inputs = node_feature.reshape((self.env.node_num, N_ROBOTS + 1))
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
            node_feature = node_feature / self.env.graph_generator.max_dist

            # transfer to node inputs tensor
            n_nodes = node_coords.shape[0]
            node_feature_inputs = node_feature.reshape((n_nodes, N_ROBOTS + 1))
            node_inputs = node_feature_inputs

            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 4)

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
            edge_for_select_node = edge_inputs

            adjacent_matrix = self.env.adjacent_matrix
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
        else:
            # get observations
            graph = copy.deepcopy(self.env.graph)
            node_feature = copy.deepcopy(self.env.node_feature)
            
            # normalize observations
            node_feature = node_feature / np.max(self.env.network_adjacent_matrix)
            
            # transfer to node inputs tensor
            node_feature_inputs = node_feature.reshape((self.env.node_num, N_ROBOTS + 1))
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

    def select_node(self, robot_index, observations, edge_for_select_node):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        next_node = self.env.reference_policy[robot_index, int(self.env.target_index), int(self.env.robot_indexes[0]), int(self.env.robot_indexes[1])]
        next_node_index = edge_for_select_node[self.env.robot_indexes[robot_index]].index(next_node)
        rp = next_node_index
        rp = torch.tensor(rp).unsqueeze(0).unsqueeze(0)

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
        return next_position, action_index, rp
    
    def select_node_shortest(self, robot_index, observations, edge_for_select_node):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations

        next_node = self.env.next_node[current_index][self.env.target_index]
        next_node_index = edge_for_select_node[self.env.robot_indexes[robot_index]].index(next_node)
        rp = next_node_index
        rp = torch.tensor(rp).unsqueeze(0).unsqueeze(0)

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
        return next_position, action_index, rp
    
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
                previous_index = copy.deepcopy(self.env.robot_indexes)
                initial_all_observations, _ = self.get_all_observations()
                action_index_list = []
                rp_list = []
                self.save_observations(initial_all_observations)
                for robot_index in range(N_ROBOTS):
                    observations, edge_for_select_node = self.get_observations(robot_index)
                    next_positions, action_index, rp = self.select_node(robot_index, observations, edge_for_select_node)

                    action_index_list.append(action_index)
                    rp_list.append(rp)

                    _, _, self.robot_positions = self.env.step(next_positions, robot_index)

                # target move
                self.env.step(None, N_ROBOTS+1, previous_index)
                done = self.env.check_done(self.env.robot_indexes)
                if done:
                    reward = 30

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

                if done:
                    break

            # save metrics
            self.perf_metrics['success_rate'] = done
            self.perf_metrics['total_steps'] = i

            # save gif
            if self.save_image:
                path = gifs_path
                self.make_gif(path, curr_episode)
        else:
            done = False
            reward = 0
            for i in range(128):
                previous_index = copy.deepcopy(self.env.robot_indexes)
                initial_all_observations, _ = self.get_all_observations()
                action_index_list = []
                rp_list = []
                self.save_observations(initial_all_observations)
                for robot_index in range(N_ROBOTS):
                    observations, edge_for_select_node = self.get_observations(robot_index)
                    next_index, action_index, rp = self.select_node(robot_index, observations, edge_for_select_node)

                    action_index_list.append(action_index)
                    rp_list.append(rp)

                    self.env.step(next_index, robot_index)
                # target move
                self.env.step(None, N_ROBOTS+1, previous_index)
                done = self.env.check_done(self.env.robot_indexes)
                if done:
                    reward = 30

                self.save_action(action_index_list)
                self.save_reference_policy(rp_list)
                self.save_reward_done(reward, done)

                final_all_observations, _ = self.get_all_observations()
                self.save_next_observations(final_all_observations)

                if done:
                    break

            self.perf_metrics['success_rate'] = done
            self.perf_metrics['total_steps'] = i

    def run_episode_shortest(self, curr_episode):
        if self.env.input_type == 'map':
            done = False
            reward = 0
            for i in range(128):
                previous_index = copy.deepcopy(self.env.robot_indexes)
                initial_all_observations, _ = self.get_all_observations()
                action_index_list = []
                rp_list = []
                self.save_observations(initial_all_observations)
                for robot_index in range(N_ROBOTS):
                    observations, edge_for_select_node = self.get_observations(robot_index)
                    next_positions, action_index, rp = self.select_node_shortest(robot_index, observations, edge_for_select_node)

                    action_index_list.append(action_index)
                    rp_list.append(rp)

                    _, _, self.robot_positions = self.env.step(next_positions, robot_index)

                # target move
                self.env.step(None, N_ROBOTS+1, previous_index)
                done = self.env.check_done(self.env.robot_indexes)
                if done:
                    reward = 30

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

                if done:
                    break

            # save metrics
            self.perf_metrics['success_rate'] = done
            self.perf_metrics['total_steps'] = i

            # save gif
            if self.save_image:
                path = gifs_path
                self.make_gif(path, curr_episode)
        else:
            done = False
            reward = 0
            for i in range(128):
                previous_index = copy.deepcopy(self.env.robot_indexes)
                initial_all_observations, _ = self.get_all_observations()
                action_index_list = []
                rp_list = []
                self.save_observations(initial_all_observations)
                for robot_index in range(N_ROBOTS):
                    observations, edge_for_select_node = self.get_observations(robot_index)
                    next_index, action_index, rp = self.select_node_shortest(robot_index, observations, edge_for_select_node)

                    action_index_list.append(action_index)
                    rp_list.append(rp)

                    self.env.step(next_index, robot_index)
                # target move
                self.env.step(None, N_ROBOTS+1, previous_index)
                done = self.env.check_done(self.env.robot_indexes)
                if done:
                    reward = 30

                self.save_action(action_index_list)
                self.save_reference_policy(rp_list)
                self.save_reward_done(reward, done)

                final_all_observations, _ = self.get_all_observations()
                self.save_next_observations(final_all_observations)

                if done:
                    break

            self.perf_metrics['success_rate'] = done
            self.perf_metrics['total_steps'] = i

    def work(self, currEpisode):
        self.run_episode(currEpisode)
        # self.run_episode_shortest(currEpisode)

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



