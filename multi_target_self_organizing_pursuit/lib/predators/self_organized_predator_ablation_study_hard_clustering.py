"""
self_organized_predator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Author: Lijun SUN.
Date: SAT APR 25 2021.
"""
import copy
import numpy as np

from lib.agents.matrix_agent import MatrixAgent
from lib.predators.clustering_ablation_study_hard_clustering import ClusteringHardClustering as Clustering
from lib.predators.ccrpursuer import CCRPursuer
from lib.predators.rl_pursuer import RLPursuer
from lib.predators.fish_schooling_searcher import FishSchoolingSearcher
from lib.predators.rl_searcher import RLSearcher

# For testing.
from lib.environment.matrix_world import MatrixWorld


class SelfOrganizedPredatorHardClustering(MatrixAgent):
    def __init__(self, env, idx_predator, under_debug=False):
        super(SelfOrganizedPredatorHardClustering, self).__init__(env, idx_predator, under_debug)

        self.clustering = Clustering(self.fov_scope)

        self.role = None
        self.cluster_center = None
        self.cluster_other_members = None

        # self.searcher = FishSchoolingSearcher(self.fov_scope, under_debug)
        self.searcher = RLSearcher(self.fov_scope, under_debug)

        self.pursuer = CCRPursuer(self.fov_scope, under_debug)
        # self.pursuer = RLPursuer(self.fov_scope, under_debug)

        self.captured_targets_map = np.zeros((env.world_rows, env.world_columns))
        self.locked_pursuers_map = np.zeros((env.world_rows, env.world_columns))

    def set_is_prey_or_not(self, true_false=False):
        self.is_prey = true_false

    def policy(self):
        # 1. Clustering.
        self.role, self.cluster_center, self.cluster_other_members, \
            self.captured_targets_map, self.locked_pursuers_map = \
            self.clustering.hard_clustering(copy.deepcopy(self.local_env_vectors),
                                            self.local_env_matrix.copy(),
                                            self.offsets_from_local_positions_to_global_positions.copy(),
                                            self.captured_targets_map.copy(),
                                            self.locked_pursuers_map.copy())

        # 2. Search.
        if self.role is "searcher":
            # next_action = self.repel_by_neighborhood()
            next_action = self.searcher.get_action(self.local_env_matrix.copy(),
                                                   copy.deepcopy(self.local_env_vectors),
                                                   self.global_own_position.copy(),
                                                   self.captured_targets_map.copy(),
                                                   self.locked_pursuers_map.copy())

            return next_action

        # 3. Single-prey pursuit.
        next_action = \
            self.pursuer.get_action(self.local_env_matrix.copy(),
                                    copy.deepcopy(self.local_env_vectors),
                                    self.cluster_center,
                                    self.cluster_other_members,
                                    self.memory["last_local_position"])
        # next_action = self.pursuer.get_action(self.local_env_matrix.copy(),
        #                                       copy.deepcopy(self.local_env_vectors),
        #                                       self.global_own_position.copy(),
        #                                       self.captured_targets_map.copy(),
        #                                       self.locked_pursuers_map.copy())

        return next_action

    def search_or_not(self):
        yes_no = False
        if self.local_own_position.tolist() == self.cluster_center.tolist():
            yes_no = True

        return yes_no


def test():
    world_rows = 40
    world_columns = 40

    n_prey = 4
    n_predators = 4 * (n_prey + 1)

    env = MatrixWorld(world_rows, world_columns,
                      n_preys=n_prey, n_predators=n_predators)
    env.reset(set_seed=True, seed=0)

    predator = SelfOrganizedPredator(env=env,
                                     idx_predator=14,
                                     under_debug=False)

    print("Step 0...")
    env.render(is_display=True, interval=0.5,
               is_save=True, is_fixed_size=False,
               grid_on=True, tick_labels_on=True,
               show_predator_idx=True,
               show_prey_idx=True)

    print("Predator index:", predator.idx_agent)
    print("Local view:", predator.local_env_vectors)
    print("Predator next action:", predator.get_action())
    print("Cluster center:", predator.cluster_center)
    print("Cluster members:", predator.cluster_other_members)
    print("Searcher next action:", predator.get_action())
    pass


if __name__ == "__main__":
    test()
