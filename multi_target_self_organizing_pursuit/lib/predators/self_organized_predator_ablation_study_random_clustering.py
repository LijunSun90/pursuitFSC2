"""
self_organized_predator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: SAT APR 25 2021.
"""
import copy
import numpy as np

from lib.agents.matrix_agent import MatrixAgent
from lib.predators.clustering_ablation_study_random_clustering import ClusteringRandomClustering as Clustering
from lib.predators.ccrpursuer import CCRPursuer
from lib.predators.rl_pursuer import RLPursuer
from lib.predators.fish_schooling_searcher import FishSchoolingSearcher
from lib.predators.rl_searcher_ablation_study_clustering_no_memory import RLSearcherClusteringNoMemory as RLSearcher

# For testing.
from lib.environment.matrix_world import MatrixWorld


class SelfOrganizedPredatorRandomClustering(MatrixAgent):
    def __init__(self, env, idx_predator, under_debug=False):
        super(SelfOrganizedPredatorRandomClustering, self).__init__(env, idx_predator, under_debug)

        self.clustering = Clustering(self.fov_scope)

        self.role = None
        self.cluster_center = None
        self.cluster_other_members = None

        # self.searcher = FishSchoolingSearcher(self.fov_scope, under_debug)
        self.searcher = RLSearcher(self.fov_scope, under_debug)

        self.pursuer = CCRPursuer(self.fov_scope, under_debug)
        # self.pursuer = RLPursuer(self.fov_scope, under_debug)

    def set_is_prey_or_not(self, true_false=False):
        self.is_prey = true_false

    def policy(self):
        # 1. Clustering.
        self.role, self.cluster_center, self.cluster_other_members = \
            self.clustering.random_clustering(copy.deepcopy(self.local_env_vectors),
                                              self.local_env_matrix.copy())

        # 2. Search.
        if self.role is "searcher":
            # next_action = self.repel_by_neighborhood()
            next_action = self.searcher.get_action(self.local_env_matrix.copy(),
                                                   copy.deepcopy(self.local_env_vectors),
                                                   self.global_own_position.copy())

            return next_action

        # 3. Single-prey pursuit.
        next_action = \
            self.pursuer.get_action(self.local_env_matrix.copy(),
                                    copy.deepcopy(self.local_env_vectors),
                                    self.cluster_center,
                                    self.cluster_other_members,
                                    self.memory["last_local_position"])

        return next_action

    def search_or_not(self):
        yes_no = False
        if self.local_own_position.tolist() == self.cluster_center.tolist():
            yes_no = True

        return yes_no

