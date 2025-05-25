import ray
import numpy as np
import os
import torch

from model import PolicyNet
from test_worker import TestWorker
from test_parameter import *

def run_test():
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    if device == 'cuda':
        checkpoint = torch.load(f'{model_path}/policy.pth')
    else:
        checkpoint = torch.load(f'{model_path}/policy.pth', map_location = torch.device('cpu'))

    global_network.load_state_dict(checkpoint)
    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    curr_test = 0

    perf_metrics_list = {'determin_worst_u': [], 'stoch_worst_u': []}

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(perf_metrics_list['determin_worst_u']) < curr_test:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                determin_metrics, stoch_metrics, info = job
                perf_metrics_list['determin_worst_u'].append(determin_metrics['evader_shortest_worst_u'])
                perf_metrics_list['stoch_worst_u'].append(stoch_metrics['evader_shortest_worst_u'])
                # perf_metrics_list['heuristic_u'].append(metrics['heuristic_u'])

            if curr_test < NUM_TEST:
                job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                curr_test += 1
        
        avg_deter_worst_u = sum(perf_metrics_list['determin_worst_u'])/len(perf_metrics_list['determin_worst_u'])
        avg_stoch_worst_u = sum(perf_metrics_list['stoch_worst_u'])/len(perf_metrics_list['stoch_worst_u'])
        print('|#Total Test:', NUM_TEST)
        print('|#Worst-Case Utility under Deterministic Policy:', avg_deter_worst_u)
        print('|#Worst-Case Utility under Stochastic Policy:', avg_stoch_worst_u)
        print('|#Worst-Case Utility on Average:', (avg_deter_worst_u + avg_stoch_worst_u)/2)

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number):
        determin_worker = TestWorker(self.meta_agent_id, self.local_network, episode_number, test=True, device=self.device, save_image=SAVE_GIFS, greedy=True, random_seed=RANDOM_SEED)
        determin_worker.work(episode_number)

        determin_metrics = determin_worker.perf_metrics

        stoch_worker = TestWorker(self.meta_agent_id, self.local_network, episode_number, test=True, device=self.device, save_image=SAVE_GIFS, greedy=False, random_seed=RANDOM_SEED)
        stoch_worker.work(episode_number)

        stoch_metrics = stoch_worker.perf_metrics
        return determin_metrics, stoch_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_weights(weights)

        determin_metrics, stoch_metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return determin_metrics, stoch_metrics, info


if __name__ == '__main__':
    ray.init(num_gpus=2)
    for i in range(NUM_RUN):
        run_test()
