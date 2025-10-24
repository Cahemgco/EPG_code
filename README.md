# Equilibrium Policy Generalization: A Reinforcement Learning Framework for Cross-Graph Zero-Shot Generalization in Pursuit-Evasion Games

This repository contains an implementation of Equilibrium Policy Generalization (EPG) proposed in our paper [1] at NeurIPS 2025. Four no-exit scenarios (2vs1-pursuer/2vs1-evader/6vs1-pursuer/6vs1-evader) and one multi-exit scenario (5vs1-pursuer) are considered in our implementation. A more complete version (with DP implementation and test files) can be found in the Supplementary Material of our paper. If you have any questions, please contact lurunyu17@mails.ucas.ac.cn.

[1] Runyu Lu, Peng Zhang, Ruochuan Shi, Yuanheng Zhu, Dongbin Zhao, Yang Liu, Dong Wang, and Cesare Alippi. Equilibrium policy generalization: A reinforcement learning framework for cross-graph zero-shot generalization in pursuit-evasion games. In Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

## Code Environment
- **Python Version**: 3.10  
- **CUDA Version**: 12.2
- **OS Version**: Ubuntu 22.04

### Main Installed Libraries

| Library         | Version  |
|-----------------|----------|
| numpy           | 1.24.0   |
| networkx        | 3.3      |
| ray             | 2.43.0   |
| scikit-image    | 0.25.2   |
| scikit-learn    | 1.6.1    |
| tensorboard     | 2.19.0   |
| torch (PyTorch) | 1.13.1   |
| matplotlib      | 3.10.1   |

## Scenarios
1. No-exit graphs (requiring equilibrium policy preprocessing)

    * 2vs1-evader (train the evader policy against 2 pursuers)
    * 2vs1-pursuer (train the 2-agent pursuer policy)
    * 6vs1-evader 
    * 6vs1-pursuer

2. Multi-exit graphs

    * 5vs1-pursuer (train the 5-agent pursuer policy under the 8-exit scenario)

## How to run

### Preprocessing in no-exit graphs
1. To preprocess array $D$ (the minimum pursuit steps under optimal pure strategies), please use the following commands:
```
cd data
python preprocess_D.py
```

2. To preprocess equilibrium policies, please use the following commands:
```
cd data
python preprocess_policy.py
```

### Train
Please set `train_mode = True` in `parameter.py`.
1. No-exit graphs (2vs1-pursuer for instance)
```
cd no-exit/2vs1-pursuer
python driver.py
```
2. Multi-exit graphs
```
cd multi-exit/5vs1-pursuer
python driver_sample.py
```

### Test
Please set `train_mode = False` in `parameter.py`.
1. Test our current model under no-exit scenarios (2vs1-pursuer for instance).
```
cd no-exit/2vs1-pursuer
python test_driver.py
```
You can choose the testing map by setting `TEST_MAP` in `test_parameter.py` to be `'Grid'`, `'ScotlandYard'`, `Downtown`, `'TimesSquare'`, `'Hollywood'`, `'Sagrada'`, `'Bund'`, `'Eiffel'`, `'BigBen'`, `'Sydney'`, or `None` (i.e., under unseen Dungeon maps).

2. Test our current model under the scenario of 5 pursuers, 1 evader and 8 exits.
```
cd multi-exit/5vs1-pursuer
python test_driver.py
```
You can choose the testing map by setting `TEST_MAP` to be `'Grid'`, `'ScotlandYard'`, or `None` (i.e., under unseen Dungeon maps).

3. To test the model trained by yourself, please first extract the policy model from your checkpoint (located at `model_path` set in `parameter.py`): 
```
checkpoint = torch.load('checkpoint.pth')
policy = checkpoint['policy_model']
torch.save(policy, 'policy.pth')
```
Then, you can replace our current model (at `pursuer_model` or `evader_model` directory) with the new policy model.
