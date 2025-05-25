from skimage import io
import matplotlib.pyplot as plt
import os
from skimage.measure import block_reduce
import random
import networkx as nx
import matplotlib.pyplot as plt
import torch

from graph_generator import *
from parameter import *
if not train_mode:
    from test_parameter import *

np.set_printoptions(threshold=np.inf)

class Env():
    def __init__(self, map_index, num_robots, random_seed=None, k_size=20, plot=False, test=False, input_type=INPUT_TYPE):
        # import environment ground truth from dungeon files
        self.test = test
        self.input_type = input_type
        if self.input_type == 'map':
            if self.test:
                self.map_dir = f'../../data/map/test'  # change to 'complex', 'medium', and 'easy'
            else:
                self.map_dir = f'../../data/map/train'
            self.random_seed = random_seed
            self.map_list = os.listdir(self.map_dir)
            self.map_list.sort(reverse=False)
            self.map_index = map_index % np.size(self.map_list)
            self.ground_truth = self.import_ground_truth(self.map_dir + '/' + self.map_list[self.map_index])

            self.ground_truth_size = np.shape(self.ground_truth)  # (480, 640)
            self.num_robots = num_robots

            # initialize graph generator
            self.graph_generator = Graph_generator(map_size=self.ground_truth_size, k_size=k_size, plot=plot, test=self.test)
            if FIXED_OPPONENT == False:
                self.robot_positions, self.robot_indexes, self.target_position, self.target_index, self.node_coords, self.graph, self.node_feature, self.reference_policy, self.adjacent_matrix, self.network_adjacent_matrix, self.opponent_policy = None, None, None, None, None, None, None, None, None, None, None
            else:
                self.robot_positions, self.robot_indexes, self.target_position, self.target_index, self.node_coords, self.graph, self.node_feature, self.reference_policy, self.adjacent_matrix, self.network_adjacent_matrix = None, None, None, None, None, None, None, None, None, None
            self.stepi = 0
            self.node_num = 0

            self.begin()
            self.robot_positions = copy.deepcopy(self.start_robot_positions)
            self.robot_indexes = copy.deepcopy(self.start_robot_indexes)
            self.target_position = copy.deepcopy(self.start_target_position)
            self.target_index = copy.deepcopy(self.start_target_index)

            k_size = self.node_coords.shape[0]

            # plot related
            self.plot = plot
            self.frame_files = []
            self.points = {}
            if self.plot:
                # initialize the route
                for i in range(self.num_robots):
                    self.points['x'+str(i+1)] = [self.start_robot_positions[i][0]]
                    self.points['y'+str(i+1)] = [self.start_robot_positions[i][1]]

        else:
            self.random_seed = random_seed
            self.num_robots = num_robots  
            if self.test:
                if TEST_MAP == None:
                    self.adj_dir = '../../data/adj_file/Dungeon_test'
                    self.adj_list = os.listdir(self.adj_dir)
                    self.adj_list.sort(reverse=False)
                    self.adj_index = map_index % np.size(self.adj_list)
                    file_path = self.adj_dir + '/' + self.adj_list[self.adj_index]

                elif TEST_MAP == 'Grid':
                    file_path = '../../data/adj_file/Grid/adj_matrix_0.txt'
                    self.adj_index = map_index % 1000

                elif TEST_MAP == 'ScotlandYard':
                    file_path = '../../data/adj_file/ScotlandYard/adj_matrix_0.txt'
                    self.adj_index = map_index % 1000

                else:
                    raise ValueError("TEST_MAP does not exist.")

            else:
                self.adj_dir = '../../data/adj_file/Dungeon_train'
                self.adj_list = os.listdir(self.adj_dir)
                self.adj_list.sort(reverse=False)
                self.adj_index = map_index % np.size(self.adj_list)
                file_path = self.adj_dir + '/' + self.adj_list[self.adj_index]

            self.graph, self.node_num, self.adjacent_matrix, self.network_adjacent_matrix, self.node_feature, self.next_node, self.start_robot_indexes, self.start_target_index, self.exit_indexes = self.import_adj_matrix(file_path, self.adj_index)
            self.robot_indexes = []
            for k in range(num_robots):
                self.robot_indexes.append(self.start_robot_indexes[k])
            self.target_index = self.start_target_index

            self.stepi = 0

    def find_index_from_coords(self, position):
        index = np.argmin(np.linalg.norm(self.node_coords - position, axis=1))
        assert self.node_num == np.linalg.norm(self.node_coords - position, axis=1).shape[0]
        return index
    
    def begin(self):
        if EXIT_NUM > 0:
            self.start_robot_positions, self.start_robot_indexes, self.start_target_position, self.start_target_index, self.node_coords, self.graph, self.node_feature, self.start_robot_indexes, self.start_target_index, self.adjacent_matrix, self.network_adjacent_matrix, self.next_node, self.exit_indexes = self.graph_generator.generate_graph(
                self.ground_truth, self.random_seed)
            
        else:
            self.robot_positions, self.robot_indexes, self.target_position, self.target_index, self.node_coords, self.graph, self.node_feature, self.robot_indexes, self.target_index, self.adjacent_matrix, self.network_adjacent_matrix = self.graph_generator.generate_graph(
                self.ground_truth, self.random_seed)
        self.edge_mask = torch.from_numpy(self.adjacent_matrix).float().unsqueeze(0)
        self.node_num = self.node_coords.shape[0]
   
    def resetenv(self):
        self.robot_indexes = copy.deepcopy(self.start_robot_indexes)
        self.target_index = copy.deepcopy(self.start_target_index)
        self.stepi = 0
        if self.input_type == 'map':
            self.robot_positions = copy.deepcopy(self.start_robot_positions)
            self.target_position = copy.deepcopy(self.start_target_position)
            self.frame_files = []
            self.node_feature = self.graph_generator.update_graph(self.robot_indexes, self.target_index)
            self.points = {}
            if self.plot:
                # initialize the route
                for i in range(self.num_robots):
                    self.points['x'+str(i+1)] = [self.start_robot_positions[i][0]]
                    self.points['y'+str(i+1)] = [self.start_robot_positions[i][1]]

        else:
            node_feature = []
            for index in range(self.node_num):
                feature = []
                feature.append(self.network_adjacent_matrix[index][self.target_index])
                for robot in range(self.num_robots):
                    feature.append(self.network_adjacent_matrix[index][robot])
                if EXIT_NUM > 0:
                    for exit in self.exit_indexes:
                        feature.append(self.network_adjacent_matrix[index][exit])
                # print(index, feature)
                node_feature.append(feature)
            node_feature = np.array(node_feature)
       
    def step(self, next_position, robot_index):
        if self.input_type == 'map':
            if robot_index <= N_ROBOTS:
                next_node_index = self.find_index_from_coords(next_position)
                self.robot_positions[robot_index] = next_position
                self.robot_indexes[robot_index] = next_node_index

                if self.plot:
                    self.points['x'+str(robot_index+1)].append(self.robot_positions[robot_index][0])
                    self.points['y'+str(robot_index+1)].append(self.robot_positions[robot_index][1])
            
                # check if done
                done, escaped = self.check_done(self.robot_indexes)
                reward = 0
                if self.test:
                    if escaped:
                        reward = -1
                    if done:
                        reward = 1
                    # else:
                    #     reward = 0

                else:
                    if escaped:
                        reward = -1
                    if done:
                        reward = 0

                # update the graph
                self.node_feature = self.graph_generator.update_graph(self.robot_indexes, self.target_index)

                return reward, done, self.robot_positions
            else:
                if FIXED_OPPONENT == False:
                    self.target_index = next_position
                    self.target_position = self.node_coords[self.target_index]
                self.stepi += 1

                # update the graph
                self.node_feature = self.graph_generator.update_graph(self.robot_indexes, self.target_index)
        else:
            if robot_index <= N_ROBOTS:
                next_index = next_position
                self.robot_indexes[robot_index] = int(next_index)
                node_feature = []

                for index in range(self.node_num):
                    feature = []
                    feature.append(self.network_adjacent_matrix[index][self.target_index])
                    for robot in range(self.num_robots):
                        feature.append(self.network_adjacent_matrix[index][self.robot_indexes[robot]])
                    if EXIT_NUM > 0:
                        for exit in self.exit_indexes:
                            feature.append(self.network_adjacent_matrix[index][exit])
                    # print(index, feature)
                    node_feature.append(feature)
                self.node_feature = np.array(node_feature)
                
            else:
                self.target_index = next_position
                self.stepi += 1
                node_feature = []
                for index in range(self.node_num):
                    feature = []
                    feature.append(self.network_adjacent_matrix[index][self.target_index])
                    for robot in range(self.num_robots):
                        feature.append(self.network_adjacent_matrix[index][self.robot_indexes[robot]])
                    if EXIT_NUM > 0:
                        for exit in self.exit_indexes:
                            feature.append(self.network_adjacent_matrix[index][exit])
                    # print(index, feature)
                    node_feature.append(feature)
                self.node_feature = np.array(node_feature)

    def import_ground_truth(self, map_index):
        # occupied 1, free 255, unexplored 127
        ground_truth = (io.imread(map_index, 1) * 255).astype(int)
        ground_truth = (ground_truth > 150)
        ground_truth = ground_truth * 254 + 1

        return ground_truth
    
    def import_adj_matrix(self, file_path, index):
        with open(file_path, 'r') as f:
            node_num = int(f.readline().strip())
            adj_matrix = []
            for line in f:
                adj_matrix.append(list(map(int, line.split())))
            adj_matrix = np.array(adj_matrix)
        graph = nx.from_numpy_array(adj_matrix)
       
        if self.test == True:
            if TEST_MAP == None:
                data = np.load('init/test.npz')
                exit_indexes = list(data['exit'][index])
                robot_indexes = list(data['robot'][index])
                target_index = data['target'][index]

                network_adjacent_matrix = np.load('init/test/'+str(index)+'.npz')['dist']
                next_node = np.load('init/test/'+str(index)+'.npz')['next']

            elif TEST_MAP == 'Grid':
                data = np.load('init/grid.npz')
                exit_indexes = list(data['exit'][index])
                robot_indexes = list(data['robot'][index])
                target_index = data['target'][index]

                matrix = np.load('init/grid_matrix.npz')
                network_adjacent_matrix = matrix['dist']
                next_node = matrix['next']

            elif TEST_MAP == 'ScotlandYard':
                data = np.load('init/SY.npz')
                exit_indexes = list(data['exit'][index])
                robot_indexes = list(data['robot'][index])
                target_index = data['target'][index]

                matrix = np.load('init/SY_matrix.npz')
                network_adjacent_matrix = matrix['dist']
                next_node = matrix['next']

            else:
                raise ValueError("TEST_MAP does not exist.")
            
        else:
            data = np.load('init/train.npz')
            exit_indexes = list(data['exit'][index])
            robot_indexes = list(data['robot'][index])
            target_index = data['target'][index]

            network_adjacent_matrix = np.load('init/train/'+str(index)+'.npz')['dist']
            next_node = np.load('init/train/'+str(index)+'.npz')['next']
        assert network_adjacent_matrix.shape[0] == adj_matrix.shape[0]
        # print('file', file_path, 'index: ', index, 'exit: ', exit_indexes, 'pursuer: ', robot_indexes, 'evader: ', target_index)

        adj_matrix = 1 - adj_matrix
        
        node_feature = []
        for index in range(node_num):
            feature = []
            feature.append(network_adjacent_matrix[index][target_index])
            for robot in range(self.num_robots):
                feature.append(network_adjacent_matrix[index][robot])
            if EXIT_NUM > 0:
                for exit in exit_indexes:
                    feature.append(network_adjacent_matrix[index][exit])
            # print(index, feature)
            node_feature.append(feature)
        node_feature = np.array(node_feature)

        return graph, node_num, adj_matrix, network_adjacent_matrix, node_feature, next_node, robot_indexes, target_index, exit_indexes
    
    def free_cells(self):
        index = np.where(self.ground_truth == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def check_done(self, node_indexes):
        cnt = 0
        for k in range(self.num_robots):
            if node_indexes[k] == self.target_index:
                cnt += 1
        done = False
        if cnt >= 1:
            done = True

        escaped = False
        if EXIT_NUM > 0:
            if self.target_index in self.exit_indexes:
                escaped = True
        
        return done, escaped
    
    def evaluate_exploration_rate(self):
        rate = np.sum(self.combined_belief == 255) / np.sum(self.ground_truth == 255)
        return rate

    def calculate_new_free_area(self, old_robot_belief, robot_belief):
        old_free_area = old_robot_belief == 255
        current_free_area = robot_belief == 255
        new_free_area = (current_free_area.astype(int) - old_free_area.astype(int)) * 255
        new_free_area = np.where(new_free_area == -255, 0, new_free_area)
        return new_free_area

    def calculate_path_length(self, path):
        dist = 0
        start = path[0]
        end = path[-1]
        for index in path:
            if index == end:
                break
            dist += np.linalg.norm(self.node_coords[start] - self.node_coords[index])
            start = index
        return dist

    def plot_env(self, n, path, step):
        plt.switch_backend('agg')
        colors = ['r', 'g', 'y', 'm', 'skyblue']
        plt.cla()
        plt.suptitle('')
        plt.imshow(self.ground_truth, cmap='gray')
        plt.axis('off')
        # plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        for i in range(len(self.graph_generator.x)):
            plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'orange', zorder=1)  # plot edges will take long time
        
        for i in range(self.num_robots):    
            plt.scatter(self.points['x'+str(i+1)][-1], self.points['y'+str(i+1)][-1], c=colors[i], s=200-i*20, zorder=7)

        if EXIT_NUM > 0:
            plt.suptitle('Total step: {}'.format(self.stepi))
            # plt.savefig('{}/{}_{}_{}_samples.png'.format(path, n, step, self.state, dpi=300))
            frame = '{}/{}_{}_samples.png'.format(path, n, step)
            for exit_index in self.exit_indexes:
                exit_position = self.node_coords[exit_index]
                plt.scatter(exit_position[0], exit_position[1], c='lime', marker='*', s=250, zorder=6)
    
        else:
            plt.suptitle('Total step: {}'.format(self.stepi))
            # plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=300))
            frame = '{}/{}_{}_samples.png'.format(path, n, step)
        
        plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c='darkblue', zorder=5)
        plt.scatter(self.node_coords[self.target_index, 0], self.node_coords[self.target_index, 1], s=100, marker='s', c='c', zorder=10)
        plt.tight_layout()
        plt.savefig(frame)
        # plt.show()
        self.frame_files.append(frame)
        plt.close()
