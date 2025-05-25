import os
import subprocess
import numpy as np

generate_policy_exec = './generate_policy'
adj_file_root = 'adj_file'
preprocess_root = 'preprocess_policy'

target_subdirs = ['Dungeon_test', 'Dungeon_train']

for subdir in target_subdirs:
    subdir_path = os.path.join(adj_file_root, subdir)
    preprocess_subdir = os.path.join(preprocess_root, subdir)
    
    if not os.path.isdir(subdir_path):
        continue

    os.makedirs(preprocess_subdir, exist_ok=True)
    for adj_file in os.listdir(subdir_path):
        if not adj_file.startswith('adj_matrix_') or not adj_file.endswith('.txt'):
            continue
        
        adj_file_path = os.path.join(subdir_path, adj_file)

        with open('adj_matrix.txt', 'w') as f:
            with open(adj_file_path, 'r') as adj_f:
                f.write(adj_f.read())
        subprocess.run([generate_policy_exec])
        data_list = []
        with open('policy.txt', 'r') as file:
            for line in file:
                row = [int(x) for x in line.strip().split()]
                data_list.append(row)
        
        data_array = np.array(data_list)
        column_1 = data_array[:, 0]
        column_2 = data_array[:, 1]
        column_3 = data_array[:, 2]
        
        with open(adj_file_path, 'r') as file:
            first_line = file.readline().strip()
            num = int(first_line.split()[0])
        
        opponent_policy = column_1.reshape(num, num, num)
        pursuer_1_policy = column_2.reshape(num, num, num)
        pursuer_2_policy = column_3.reshape(num, num, num)
        
        pursuer_policy = np.array([pursuer_1_policy, pursuer_2_policy])
        index = adj_file.replace('adj_matrix_', '').replace('.txt', '')
        np.save(os.path.join(preprocess_subdir, f'opponent_policy_{index}.npy'), opponent_policy)
        np.save(os.path.join(preprocess_subdir, f'pursuer_policy_{index}.npy'), pursuer_policy)

        os.remove('adj_matrix.txt')
        os.remove('policy.txt')
        
        print(f'Finished: {subdir}/{adj_file}')