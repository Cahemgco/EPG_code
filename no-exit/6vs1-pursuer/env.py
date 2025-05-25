from skimage import io
import matplotlib.pyplot as plt
import os
from skimage.measure import block_reduce
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from itertools import permutations

# from sensor import *
from graph_generator import *
from collections import Counter

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
                self.map_dir = f'../../data/map/test'
            else:
                self.map_dir = f'../../data/map/train'
            self.random_seed = random_seed + map_index
            self.num_robots = num_robots
            self.map_list = os.listdir(self.map_dir)
            self.map_list.sort(reverse=False)
            self.map_index = map_index % np.size(self.map_list)
            self.ground_truth = self.import_ground_truth(self.map_dir + '/' + self.map_list[self.map_index])

            self.ground_truth_size = np.shape(self.ground_truth)  # (480, 640)
           
            # initialize graph generator
            self.graph_generator = Graph_generator(map_size=self.ground_truth_size, k_size=k_size, plot=plot)
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
                    self.points['x'+str(i+1)] = [self.robot_positions[i][0]]
                    self.points['y'+str(i+1)] = [self.robot_positions[i][1]]

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
                    test_dp_path = '../../data/preprocess_policy/Dungeon_test'
                    test_dist_path = '../../data/preprocess_D/Dungeon_test'

                else:
                    file_path = '../../data/adj_file/' + TEST_MAP + '/adj_matrix_0.txt'
                    self.adj_index = 0
                    test_dp_path = '../../data/preprocess_policy/' + TEST_MAP
                    test_dist_path = '../../data/preprocess_D/' + TEST_MAP

            else:
                self.adj_dir = '../../data/adj_file/Dungeon_train'
                self.adj_list = os.listdir(self.adj_dir)
                self.adj_list.sort(reverse=False)
                self.adj_index = map_index % np.size(self.adj_list)
                file_path = self.adj_dir + '/' + self.adj_list[self.adj_index]
                train_dp_path = '../../data/preprocess_policy/Dungeon_train'
                train_dist_path = '../../data/preprocess_D/Dungeon_train'

            self.graph, self.node_num, self.adjacent_matrix, self.network_adjacent_matrix, self.node_feature,\
                self.start_indexes, self.target_index, self.next_node = self.import_adj_matrix(file_path)
            self.robot_indexes = []
            for k in range(num_robots):
                self.robot_indexes.append(self.start_indexes[k])
   
            self.stepi = 0

            if self.test:
                if TEST_MAP == None:
                    self.reference_policy = np.load(str(test_dp_path) + '/pursuer_policy_'+str(self.adj_index).zfill(3)+'.npy')
                    self.dist_data = np.load(str(test_dist_path) + '/dist_'+str(self.adj_index).zfill(3)+'.npy')
                else:
                    self.reference_policy = np.load(str(test_dp_path) + '/pursuer_policy_'+str(self.adj_index)+'.npy')
                    self.dist_data = np.load(str(test_dist_path) + '/dist_'+str(self.adj_index)+'.npy')
                
            else:
                self.reference_policy = np.load(str(train_dp_path) + '/pursuer_policy_'+str(self.adj_index).zfill(3)+'.npy')
                self.dist_data = np.load(str(train_dist_path) + '/dist_'+str(self.adj_index).zfill(3)+'.npy')

            if FIXED_OPPONENT == False:
                if self.test:
                    if TEST_MAP == None:
                        self.opponent_policy = np.load(str(test_dp_path) + '/opponent_policy_'+str(self.adj_index).zfill(3)+'.npy')
                    else:
                        self.opponent_policy = np.load(str(test_dp_path) + '/opponent_policy_'+str(self.adj_index)+'.npy')
                else:
                    self.opponent_policy = np.load(str(train_dp_path) + '/opponent_policy_'+str(self.adj_index).zfill(3)+'.npy')
    
    def find_index_from_coords(self, position):
        index = np.argmin(np.linalg.norm(self.node_coords - position, axis=1))
        assert self.node_num == np.linalg.norm(self.node_coords - position, axis=1).shape[0]
        return index

    def begin(self):
        file_path = str(self.map_list[self.map_index])

        self.start_robot_positions, self.start_robot_indexes, self.start_target_position, \
            self.start_target_index, self.node_coords, self.graph, self.node_feature, self.robot_indexes, \
                self.target_index, self.adjacent_matrix, self.network_adjacent_matrix, \
                    self.next_node = self.graph_generator.generate_graph(self.ground_truth, self.map_index)
        
        self.node_num = self.node_coords.shape[0]
        
        if self.test:
            self.reference_policy = np.load(str(test_dp_path) + '/pursuer_policy_'+str(self.map_index)+'.npy')
            self.dist_data = np.load(str(test_dist_path) + '/dist_'+str(self.map_index)+'.npy')
            
        else:
            self.reference_policy = np.load(str(train_dp_path) + '/pursuer_policy_'+str(self.map_index)+'.npy')
            self.dist_data = np.load(str(train_dist_path) + '/dist_'+str(self.map_index)+'.npy')
        
        if FIXED_OPPONENT == False:
            if self.test:
                self.opponent_policy = np.load(str(test_dp_path) + '/opponent_policy_'+str(self.map_index)+'.npy')
                
            else:
                self.opponent_policy = np.load(str(train_dp_path) + '/opponent_policy_'+str(self.map_index)+'.npy')

        assert self.node_num == self.opponent_policy.shape[0], print("node_num: ", self.node_num, "self.opponent_policy: ", self.opponent_policy.shape)

    def step(self, next_position, robot_index, previous_index = None):
        if self.input_type == 'map':
            if robot_index <= N_ROBOTS:
                next_node_index = self.find_index_from_coords(next_position)
                self.robot_positions[robot_index] = next_position
                self.robot_indexes[robot_index] = next_node_index

                if self.plot:
                    self.points['x'+str(robot_index+1)].append(self.robot_positions[robot_index][0])
                    self.points['y'+str(robot_index+1)].append(self.robot_positions[robot_index][1])
            
                # check if done
                done = self.check_done(self.robot_indexes)
                reward = 0
                if done:
                    reward = 30 # a finishing reward
                # update the graph
                self.node_feature = self.graph_generator.update_graph(self.robot_indexes, self.target_index)

                return reward, done, self.robot_positions
            else:
                
                combs = []
                
                def get_paired_combinations(elements):
                    n = len(elements)
                    assert n % 2 == 0
                    
                    def helper(elements):
                        if not elements:
                            return [[]]
                        result = []
                        for i in range(1, len(elements)):
                            pair = (elements[0], elements[i])
                            remaining = elements[1:i] + elements[i+1:]
                            for rest in helper(remaining):
                                result.append([pair] + rest)
                        return result

                    all_combinations = helper(elements)
                    sorted_combinations = [tuple(sorted(tuple(sorted(pair)) for pair in combination)) 
                                        for combination in all_combinations]
                    unique_combinations = set(sorted_combinations)
                    
                    return list(unique_combinations)

                all_pairs = get_paired_combinations(previous_index)
                # print('all_pairs: ', len(all_pairs))

                max_distances = []
                max_combs = []

                for pairs in all_pairs:
                    dist1 = self.dist_data[pairs[0][0], pairs[0][1], self.target_index]
                    dist2 = self.dist_data[pairs[1][0], pairs[1][1], self.target_index]
                    dist3 = self.dist_data[pairs[2][0], pairs[2][1], self.target_index]

                    max_dist = max(dist1, dist2, dist3)
                    
                    if max_dist == dist1:
                        comb = (pairs[0])
                    elif max_dist == dist2:
                        comb = (pairs[1])
                    elif max_dist == dist3:
                        comb = (pairs[2])
                                        
                    max_distances.append(max_dist)
                    max_combs.append(comb)

                if max_distances:
                    min_dist_index = max_distances.index(min(max_distances))
                    min_dist = max_distances[min_dist_index]
                    min_comb = max_combs[min_dist_index]
                else:
                    min_dist = float('inf')
                    min_comb = None
                
                if FIXED_OPPONENT == False:
                    self.target_index = self.get_opponent_position(min_comb, self.target_index)
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
                    # print(index, feature)
                    node_feature.append(feature)
                self.node_feature = np.array(node_feature)
                
            else:
                def get_paired_combinations(elements):
                    n = len(elements)
                    assert n % 2 == 0 
                    def helper(elements):
                        if not elements:
                            return [[]]
                        result = []
                        
                        for i in range(1, len(elements)):
                            pair = (elements[0], elements[i])
                            remaining = elements[1:i] + elements[i+1:]
                            for rest in helper(remaining):
                                result.append([pair] + rest)
                        return result

                    all_combinations = helper(elements)
                    sorted_combinations = [tuple(sorted(tuple(sorted(pair)) for pair in combination)) 
                                        for combination in all_combinations]
                    unique_combinations = set(sorted_combinations)
                    
                    return list(unique_combinations)

                all_pairs = get_paired_combinations(previous_index)
                max_distances = []
                max_combs = []

                for pairs in all_pairs:
                    dist1 = self.dist_data[pairs[0][0], pairs[0][1], self.target_index]
                    dist2 = self.dist_data[pairs[1][0], pairs[1][1], self.target_index]
                    dist3 = self.dist_data[pairs[2][0], pairs[2][1], self.target_index]

                    max_dist = max(dist1, dist2, dist3)
                    
                    if max_dist == dist1:
                        comb = (pairs[0])
                    elif max_dist == dist2:
                        comb = (pairs[1])
                    elif max_dist == dist3:
                        comb = (pairs[2])
                                        
                    max_distances.append(max_dist)
                    max_combs.append(comb)

                if max_distances:
                    min_dist_index = max_distances.index(min(max_distances))
                    min_dist = max_distances[min_dist_index]
                    min_comb = max_combs[min_dist_index]
                else:
                    min_dist = float('inf')
                    min_comb = None

                self.target_index = self.get_opponent_position(min_comb, self.target_index)
                self.stepi += 1
                node_feature = []
                for index in range(self.node_num):
                    feature = []
                    feature.append(self.network_adjacent_matrix[index][self.target_index])
                    for robot in range(self.num_robots):
                        feature.append(self.network_adjacent_matrix[index][self.robot_indexes[robot]])
                    node_feature.append(feature)
                self.node_feature = np.array(node_feature)

    def import_ground_truth(self, map_index):
        # occupied 1, free 255, unexplored 127
        ground_truth = (io.imread(map_index, 1) * 255).astype(int)
        ground_truth = (ground_truth > 150)
        ground_truth = ground_truth * 254 + 1

        return ground_truth
    
    def import_adj_matrix(self, file_path):
        with open(file_path, 'r') as f:
            node_num = int(f.readline().strip())
            adj_matrix = []
            for line in f:
                adj_matrix.append(list(map(int, line.split())))
            adj_matrix = np.array(adj_matrix)
        graph = nx.from_numpy_array(adj_matrix)

        network_adjacent_matrix = nx.floyd_warshall_numpy(graph)

        next_node = [[j for j in range(node_num)] for i in range(node_num)]
        for i in range(node_num):
            for j in range(node_num):
                for k in range(node_num):
                    if network_adjacent_matrix[i][k] == 1 and network_adjacent_matrix[i][j] == network_adjacent_matrix[k][j] + 1:
                        next_node[i][j] = k
        next_node = np.array(next_node)

        if self.random_seed != None: 
            random.seed(self.random_seed)
            robot_index = random.randint(0, node_num-1)
            random.seed(self.random_seed + 10)
            target_index = random.randint(0, node_num-1)
        else:
            count = 0
            while True:
                count += 1
                robot_index = random.randint(0, node_num-1)
                target_index = random.randint(0, node_num-1)
                dist = nx.shortest_path_length(graph, source=robot_index, target=target_index)
                # print('dist: ', dist)
                if dist > 5 or count > 1000:
                    break
        robot_indexes = [robot_index, robot_index, robot_index, robot_index, robot_index, robot_index]
        adj_matrix = 1 - adj_matrix
        
        node_feature = []
        for index in range(node_num):
            feature = []
            feature.append(network_adjacent_matrix[index][target_index])
            for robot in range(self.num_robots):
                feature.append(network_adjacent_matrix[index][robot_indexes[robot]])
            # print(index, feature)
            node_feature.append(feature)
        node_feature = np.array(node_feature)

        return graph, node_num, adj_matrix, network_adjacent_matrix, node_feature, robot_indexes, target_index, next_node
    
    def free_cells(self):
        index = np.where(self.ground_truth == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def check_done(self, node_indexes):
        cnt = 0
        for k in range(self.num_robots):
            if self.adjacent_matrix[node_indexes[k]][self.target_index] == 0:
                cnt += 1
        done = False

        if cnt >= 3:
            done = True
        return done

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
        colors = ['r', 'g', 'y', 'b', 'c', 'm']
        plt.cla()
        plt.suptitle('')
        plt.imshow(self.ground_truth, cmap='gray')
        plt.axis('off')
        for i in range(len(self.graph_generator.x)):
            plt.plot(self.graph_generator.x[i], self.graph_generator.y[i], 'orange', zorder=1)  # plot edges will take long time
        for i in range(self.num_robots):
            plt.scatter(self.points['x'+str(i+1)][-1], self.points['y'+str(i+1)][-1], c='r', s=150, zorder=6)
        
        index_counts = Counter(self.robot_indexes)
        for index, count in index_counts.items():
            if count > 1:
                plt.text(self.node_coords[index][0]-5, self.node_coords[index][1]+6, str(count), fontsize=10, color='black', zorder=11)

        plt.scatter(self.node_coords[:, 0], self.node_coords[:, 1], c='darkblue', zorder=5)
        plt.suptitle('Total step: {}'.format(self.stepi))
        plt.scatter(self.node_coords[self.target_index, 0], self.node_coords[self.target_index, 1], s=60, marker='s', c='cyan', zorder=10)
        plt.tight_layout()
        
        plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=300))
        # plt.show()
        frame = '{}/{}_{}_samples.png'.format(path, n, step)
        self.frame_files.append(frame)
        plt.close()

    def get_opponent_position(self, robot_index, opponent_index):
        next_opponent_position = self.opponent_policy[int(opponent_index)][robot_index[0]][robot_index[1]]
        
        return next_opponent_position