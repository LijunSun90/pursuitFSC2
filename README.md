## Fuzzy self-organizing cooperative coevolution (FSC2) for multi-target self-organizing pursuit

Official code for the paper "Toward multi-target self-organizing pursuit in a partially observable Markov game", which has been submitted to [Arxiv](https://arxiv.org/) and Applied Soft Computing for reviewing.

![Alt Text](https://github.com/LijunSun90/pursuitFSC2/blob/main/multi_target_self_organizing_pursuit/data/case_study_40x40_4t_16p.gif)
![Alt Text](https://github.com/LijunSun90/pursuitFSC2/blob/main/multi_target_self_organizing_pursuit/data/case_study_80x80_256t_1024p.gif)

Using the following to cite:
...

### Description

In the proposed [FSC2](https://github.com/LijunSun90/pursuitFSC2/tree/main/multi_target_self_organizing_pursuit/lib/predators/self_organized_predator.py), 
the multi-target self-organizing pursuit (MTSOP or SOP) problem is decomposed into three subtasks: 
fuzzy-based distributed task allocation ([DTA](https://github.com/LijunSun90/pursuitFSC2/tree/main/multi_target_self_organizing_pursuit/lib/predators/clustering.py)),
self-organizing search ([SOS](https://github.com/LijunSun90/pursuitFSC2/tree/main/multi_target_self_organizing_pursuit/lib/predators/rl_searcher.py)), and
single-target pursuit ([STP](https://github.com/LijunSun90/pursuitFSC2/tree/main/multi_target_self_organizing_pursuit/lib/predators/ccrpursuer.py)).

- The MTSOP, i.e, the proposed FSC2 algorithm, is in the folder [multi_target_self_organizing_pursuit](https://github.com/LijunSun90/pursuitFSC2/tree/main/multi_target_self_organizing_pursuit).

- The SOS task is trained and tested in the folder [multi_target_self_organizing_search](https://github.com/LijunSun90/pursuitFSC2/tree/main/multi_target_self_organizing_search).

- The proposed [global distributed consistency (DC) metric in task allocation ](https://github.com/LijunSun90/pursuitFSC2/tree/main/multi_target_self_organizing_pursuit/fuzzy_clustering_metric/compute_consistency.py) is in the folder [fuzzy_clustering_metric](https://github.com/LijunSun90/pursuitFSC2/tree/main/multi_target_self_organizing_pursuit/fuzzy_clustering_metric).


### Dependencies tested on

Python 3.7.11

numpy 1.19.1

torch 1.10.2

mpi4py 3.1.3

To run the comparison code of ApeX-DQN, additional dependencies are:

tensorflow 1.15.0

ray 1.10.0

To run the comparison code of MADDPG, additional dependencies are:

https://github.com/openai/multiagent-particle-envs

All the dependencies are listed in the file [environment_for_mtsop_fsc2.yml](https://github.com/LijunSun90/pursuitFSC2/tree/main/multi_target_self_organizing_pursuit/environment_for_mtsop_fsc2.yml).

### Acknowledgements:
The actor-critic codes are mostly from and modified based on
- https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/vpg

The ApeX-DQN codes for comparison are from
- https://github.com/aecgames-paper/aecgames
- https://github.com/parametersharingmadrl/parametersharingmadrl

The MADDPG codes for comparison are from
- https://github.com/parametersharingmadrl/parametersharingmadrl
- https://github.com/openai/maddpg

