import os
import subprocess
import numpy as np

generate_policy_exec = './generate_D'
adj_file_root = 'adj_file'
preprocess_root = 'preprocess_D'

for subdir in os.listdir(adj_file_root):
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
        with open('D.txt', 'r') as file:
            for line in file:
                row = [int(x) for x in line.strip().split()]
                data_list.append(row)

        dist_data = np.array(data_list)
        with open(adj_file_path, 'r') as file:
            first_line = file.readline().strip()
            num = int(first_line.split()[0])

        dist_data = dist_data.reshape(num, num, num)
        index = adj_file.replace('adj_matrix_', '').replace('.txt', '')
        np.save(os.path.join(preprocess_subdir, f'dist_{index}.npy'), dist_data)

        os.remove('adj_matrix.txt')
        os.remove('D.txt')
        
        print(f'Finished: {subdir}/{adj_file}')