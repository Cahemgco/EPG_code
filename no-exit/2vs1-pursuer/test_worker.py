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
    def __init__(self, meta_agent_id, policy_net, global_step, device='cuda', greedy=False, save_image=False, random_seed=None):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image, test=True, num_robots=N_ROBOTS, random_seed=random_seed, input_type=INPUT_TYPE)
        self.local_policy_net = policy_net
        self.travel_dist = 0
        self.perf_metrics = dict()
        if self.env.input_type == 'map':
            self.robot_positions = [self.env.start_positions[i] for i in range(N_ROBOTS)]

    def run_episode(self, curr_episode):
        if self.env.input_type == 'map':
            done = False
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, 0)
            for i in range(128):
                previous_index = copy.deepcopy(self.env.robot_indexes)
                for robot_index in range(N_ROBOTS):
                    observations = self.get_observations(robot_index)
                    next_position, _ = self.select_node(observations)
                    _, _, self.robot_positions = self.env.step(next_position, robot_index)

                # target move
                self.env.step(None, N_ROBOTS+1, previous_index)
                done = self.env.check_done(self.env.robot_indexes)
                # save evaluation data
                if SAVE_TRAJECTORY:
                    if not os.path.exists(trajectory_path):
                        os.makedirs(trajectory_path)
                    csv_filename = trajectory_path + f'{curr_episode}_trajectory_result.csv'
                    new_file = False if os.path.exists(csv_filename) else True
                    field_names = ['evader', 'pursuer1', 'pursuer2']
                    with open(csv_filename, 'a') as csvfile:
                        writer = csv.writer(csvfile)
                        if new_file:
                            writer.writerow(field_names)
                        csv_data = np.array([self.env.target_index, self.env.robot_indexes[0], self.env.robot_indexes[1]]).reshape(1, -1)
                        writer.writerows(csv_data)

                # save a frame
                if self.save_image:
                    if not os.path.exists(gifs_path):
                        os.makedirs(gifs_path)
                    self.env.plot_env(self.global_step, gifs_path, i+1)
                if done:
                    break

            self.perf_metrics['success_rate'] = done
            self.perf_metrics['steps'] = i

            # save final path length
            if SAVE_LENGTH:
                if not os.path.exists(length_path):
                    os.makedirs(length_path)
                csv_filename = f'results/length/ours_length_result.csv'
                new_file = False if os.path.exists(csv_filename) else True
                field_names = ['dist']
                with open(csv_filename, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    if new_file:
                        writer.writerow(field_names)
                    csv_data = np.array([self.travel_dist]).reshape(-1,1)
                    writer.writerows(csv_data)

            # save gif
            if self.save_image:
                path = gifs_path
                self.make_gif(path, curr_episode)

        else:
            done = False
            for i in range(128):
                previous_index = copy.deepcopy(self.env.robot_indexes)
                for robot_index in range(N_ROBOTS):
                    observations = self.get_observations(robot_index)
                    next_index, _ = self.select_node(observations)
                    self.env.step(next_index, robot_index)
                    # print('robot: ', robot_index, 'next index: ', int(next_index))

                # target move
                self.env.step(None, N_ROBOTS+1, previous_index)
                done = self.env.check_done(self.env.robot_indexes)
                
                if SAVE_TRAJECTORY:
                    if not os.path.exists(trajectory_path):
                        os.makedirs(trajectory_path)
                    csv_filename = trajectory_path + f'{curr_episode}_trajectory_result.csv'
                    new_file = False if os.path.exists(csv_filename) else True
                    field_names = ['evader', 'pursuer1', 'pursuer2']
                    with open(csv_filename, 'a') as csvfile:
                        writer = csv.writer(csvfile)
                        if new_file:
                            writer.writerow(field_names)
                        csv_data = np.array([self.env.target_index, self.env.robot_indexes[0], self.env.robot_indexes[1]]).reshape(1, -1)
                        writer.writerows(csv_data)

                if done:
                    break
            self.perf_metrics['success_rate'] = done
            self.perf_metrics['steps'] = i

    def run_episode_DP(self, curr_episode):
        if self.env.input_type == 'map':
            done = False
            for i in range(128):
                previous_index = copy.deepcopy(self.env.robot_indexes)
                observations = self.get_all_observations()
                rp = self.select_node_DP(observations)
                for robot_index in range(N_ROBOTS):
                    _, _, self.robot_positions = self.env.step(rp[robot_index], robot_index)
                # target move
                self.env.step(None, N_ROBOTS+1, previous_index)
                done = self.env.check_done(self.env.robot_indexes)
                # save evaluation data
                if SAVE_TRAJECTORY:
                    if not os.path.exists(trajectory_path):
                        os.makedirs(trajectory_path)
                    csv_filename = trajectory_path + f'{curr_episode}_trajectory_result.csv'
                    new_file = False if os.path.exists(csv_filename) else True
                    field_names = ['evader', 'pursuer1', 'pursuer2']
                    with open(csv_filename, 'a') as csvfile:
                        writer = csv.writer(csvfile)
                        if new_file:
                            writer.writerow(field_names)
                        csv_data = np.array([self.env.target_index, self.env.robot_indexes[0], self.env.robot_indexes[1]]).reshape(1, -1)
                        writer.writerows(csv_data)

                # save a frame
                if self.save_image:
                    if not os.path.exists(gifs_path):
                        os.makedirs(gifs_path)
                    self.env.plot_env(self.global_step, gifs_path, i)
                if done:
                    break

            self.perf_metrics['success_rate'] = done
            self.perf_metrics['steps'] = i

            # save final path length
            if SAVE_LENGTH:
                if not os.path.exists(length_path):
                    os.makedirs(length_path)
                csv_filename = f'results/length/ours_length_result.csv'
                new_file = False if os.path.exists(csv_filename) else True
                field_names = ['dist']
                with open(csv_filename, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    if new_file:
                        writer.writerow(field_names)
                    csv_data = np.array([self.travel_dist]).reshape(-1,1)
                    writer.writerows(csv_data)

            # save gif
            if self.save_image:
                path = gifs_path
                self.make_gif(path, curr_episode)
        else:
            done = False
            for i in range(128):
                previous_index = copy.deepcopy(self.env.robot_indexes)
                for robot_index in range(N_ROBOTS):
                    next_index = self.env.reference_policy[robot_index, int(self.env.target_index), int(previous_index[0]), int(previous_index[1])]
                    self.env.robot_indexes[robot_index] = next_index

                # target move
                self.env.step(None, N_ROBOTS+1, previous_index)
                done = self.env.check_done(self.env.robot_indexes)
                if SAVE_TRAJECTORY:
                    if not os.path.exists(trajectory_path):
                        os.makedirs(trajectory_path)
                    csv_filename = trajectory_path + f'{curr_episode}_trajectory_result.csv'
                    new_file = False if os.path.exists(csv_filename) else True
                    field_names = ['evader', 'pursuer1', 'pursuer2']
                    with open(csv_filename, 'a') as csvfile:
                        writer = csv.writer(csvfile)
                        if new_file:
                            writer.writerow(field_names)
                        csv_data = np.array([self.env.target_index, self.env.robot_indexes[0], self.env.robot_indexes[1]]).reshape(1, -1)
                        writer.writerows(csv_data)

                if done:
                    break
            self.perf_metrics['success_rate'] = done
            self.perf_metrics['steps'] = i

    def run_episode_shortest(self, curr_episode):
        if self.env.input_type == 'map':
            done = False
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, 0)
            for i in range(128):
                previous_index = copy.deepcopy(self.env.robot_indexes)

                rp = self.select_node_shortest()
                for robot_index in range(N_ROBOTS):
                    _, _, self.robot_positions = self.env.step(rp[robot_index], robot_index)
                # target move
                self.env.step(None, N_ROBOTS+1, previous_index)
                done = self.env.check_done(self.env.robot_indexes)
                # save evaluation data
                if SAVE_TRAJECTORY:
                    if not os.path.exists(trajectory_path):
                        os.makedirs(trajectory_path)
                    csv_filename = trajectory_path + f'{curr_episode}_trajectory_result.csv'
                    new_file = False if os.path.exists(csv_filename) else True
                    field_names = ['evader', 'pursuer1', 'pursuer2']
                    with open(csv_filename, 'a') as csvfile:
                        writer = csv.writer(csvfile)
                        if new_file:
                            writer.writerow(field_names)
                        csv_data = np.array([self.env.target_index, self.env.robot_indexes[0], self.env.robot_indexes[1]]).reshape(1, -1)
                        writer.writerows(csv_data)

                # save a frame
                if self.save_image:
                    if not os.path.exists(gifs_path):
                        os.makedirs(gifs_path)
                    self.env.plot_env(self.global_step, gifs_path, i+1)
                if done:
                    break

            self.perf_metrics['success_rate'] = done
            self.perf_metrics['steps'] = i

            # save final path length
            if SAVE_LENGTH:
                if not os.path.exists(length_path):
                    os.makedirs(length_path)
                csv_filename = f'results/length/ours_length_result.csv'
                new_file = False if os.path.exists(csv_filename) else True
                field_names = ['dist']
                with open(csv_filename, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    if new_file:
                        writer.writerow(field_names)
                    csv_data = np.array([self.travel_dist]).reshape(-1,1)
                    writer.writerows(csv_data)

            # save gif
            if self.save_image:
                path = gifs_path
                self.make_gif(path, curr_episode)
        else:
            done = False
            for i in range(128):
                previous_index = copy.deepcopy(self.env.robot_indexes)
                rp = self.select_node_shortest()
                for robot_index in range(N_ROBOTS):
                    self.env.step(rp[robot_index], robot_index)

                # target move
                self.env.step(None, N_ROBOTS+1, previous_index)
                done = self.env.check_done(self.env.robot_indexes)
                if SAVE_TRAJECTORY:
                    if not os.path.exists(trajectory_path):
                        os.makedirs(trajectory_path)
                    csv_filename = trajectory_path + f'{curr_episode}_trajectory_result.csv'
                    new_file = False if os.path.exists(csv_filename) else True
                    field_names = ['evader', 'pursuer1', 'pursuer2']
                    with open(csv_filename, 'a') as csvfile:
                        writer = csv.writer(csvfile)
                        if new_file:
                            writer.writerow(field_names)
                        csv_data = np.array([self.env.target_index, self.env.robot_indexes[0], self.env.robot_indexes[1]]).reshape(1, -1)
                        writer.writerows(csv_data)

                if done:
                    break
            self.perf_metrics['success_rate'] = done
            self.perf_metrics['steps'] = i


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

            # get the node index of the current robot position
            current_node_index = self.env.find_index_from_coords(self.robot_positions[robot_index])
            current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)

            # calculate a mask for padded node
            node_padding_mask = None

            # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
            graph = list(graph.values())
            edge_inputs = []
            for node in graph:
                node_edges = list(map(int, node))
                edge_inputs.append(node_edges)

            adjacent_matrix = self.env.adjacent_matrix

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
            node_feature_inputs = node_feature.reshape((self.env.node_num, N_ROBOTS + 1))
            node_inputs = node_feature_inputs
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 4)

            # get the node index of the current robot position
            current_node_index = self.env.robot_indexes[robot_index]
            current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)

            # calculate a mask for padded node
            node_padding_mask = None

            # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
            edge_inputs = []
            for node in graph.nodes():
                neighbors = list(graph.neighbors(node))
                edge_inputs.append(neighbors)

            adjacent_matrix = self.env.adjacent_matrix
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
        node_feature_inputs = node_feature.reshape((n_nodes, N_ROBOTS + 1))
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
        
        assert self.env.adjacent_matrix[next_node_index][current_index] == 0, print('current: ', current_index, 'edge_inputs: ', edge_inputs, 'next: ', next_node_index)
        return next_position, action_index
    
    def select_node_DP(self, observations):
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask= observations
        next_node_1 = self.env.reference_policy[0, int(self.env.target_index), int(current_index.flatten()[0]), int(current_index.flatten()[1])]
        next_node_2 = self.env.reference_policy[1, int(self.env.target_index), int(current_index.flatten()[0]), int(current_index.flatten()[1])]
        
        rp = [self.env.node_coords[next_node_1], self.env.node_coords[next_node_2]]
        return rp
    
    def select_node_shortest(self):
        rp = []
        for robot_index in range(N_ROBOTS):
            next_node = self.env.next_node[self.env.robot_indexes[robot_index]][self.env.target_index]
            assert self.env.adjacent_matrix[next_node][self.env.robot_indexes[robot_index]] == 0
            if self.env.input_type == 'map':
                next_position = self.env.node_coords[next_node]
            else:
                next_position = next_node
            rp.append(next_position)

        return rp

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
        if POLICY_TYPE == 'RL':
            self.run_episode(curr_episode)
        else:
            self.run_episode_DP(curr_episode)
            # self.run_episode_shortest(curr_episode)