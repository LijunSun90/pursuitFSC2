"""
clustering.py
~~~~~~~~~~~~~

Author: Lijun SUN.
Date: FRI APR 25 2021.
"""
import copy
import numpy as np


class ClusteringHardClustering:
    def __init__(self, fov_scope):
        self.fov_scope = fov_scope

        self.fov_radius = int(0.5 * (self.fov_scope - 1))

        # Relative position in one agent's local view.
        # If fov_scope = 7, own_position = [3, 3].
        self.own_position = np.array([self.fov_radius] * 2)

        # Other parameters.
        # N, E, S, W.
        self.axial_neighbors_mask = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        self.captured_targets_map = None
        self.locked_pursuers_map = None
        pass

    def hard_clustering(self, local_env_vectors, local_env_matrix, offsets_from_local_positions_to_global_positions,
                        captured_targets_map, locked_pursuers_map):
        """
        Similar to k-means that each agent belongs to the nearest cluster center.
        Different from k-means that cluster centers are known (evaders) that do not need to be calculated iteratively.
        """
        local_env_vectors = copy.deepcopy(local_env_vectors)

        # Set of data.
        local_predators = local_env_vectors["local_predators"]

        # Cluster centers.
        local_preys = local_env_vectors["local_preys"]

        self.captured_targets_map = captured_targets_map
        self.locked_pursuers_map = locked_pursuers_map

        # Remove already captured preys from the candidate cluster centers.
        valid_index = []
        exist_axial_neighboring_preys = False
        axial_neighboring_prey = None
        for idx, prey in enumerate(local_preys):
            if not self.is_captured(prey, local_env_matrix, offsets_from_local_positions_to_global_positions):
                valid_index.append(idx)

            offset = np.abs(self.own_position - prey).tolist()
            if offset == [0, 1] or offset == [1, 0]:
                exist_axial_neighboring_preys = True
                axial_neighboring_prey = prey

        local_free_preys = local_preys[valid_index, :]

        # Remove locked pursuers to avoid taking locked pursuers as cluster members, which is wrong.
        valid_index = []
        for idx, pursuer in enumerate(local_predators):
            pursuer_global_position = pursuer + offsets_from_local_positions_to_global_positions
            is_locked_pursuer = self.locked_pursuers_map[pursuer_global_position[0], pursuer_global_position[1]]
            if not is_locked_pursuer:
                valid_index.append(idx)

        local_free_predators = local_predators[valid_index, :]

        # No local preys, the predator itself forms a single cluster.
        # Avoid oscillation between roles of pursuer and searcher,
        # which may happen when the agent is a pursuer a step nearer to the target and a searcher a step farther.
        if local_free_preys.shape[0] == 0:

            cluster_other_members = local_free_predators
            if exist_axial_neighboring_preys:
                cluster_center = axial_neighboring_prey
                role = "pursuer"
            else:
                cluster_center = self.own_position
                role = "searcher"

            return role, cluster_center, cluster_other_members, \
                self.captured_targets_map.copy(), self.locked_pursuers_map.copy()

        role = "pursuer"

        local_free_predators = np.vstack((self.own_position, local_free_predators))

        # No. of data.
        n_data = len(local_free_predators)

        # No. of clusters.
        n_clusters = len(local_free_preys)

        # Membership function.
        mu_matrix = np.zeros((n_data, n_clusters))

        # Cluster identification.
        cluster_id_matrix = np.zeros((n_data,), dtype=int)

        for i_data in range(n_data):
            mu_i = np.zeros((n_clusters,))

            for j_cluster in range(n_clusters):
                # Euclidean distance.
                # mu_i[j_cluster] = \
                #     np.sum((predators[i_data] - preys[j_cluster]) ** 2) \
                #     ** (1/(1-m))

                fov_distance = np.linalg.norm(local_free_predators[i_data] - local_free_preys[j_cluster], ord=np.inf)
                if fov_distance > self.fov_radius:
                    mu_i[j_cluster] = np.inf
                else:
                    # Manhattan distance.
                    mu_i[j_cluster] = np.linalg.norm(
                        (local_free_predators[i_data] - local_free_preys[j_cluster]),
                        ord=1)

            mu_matrix[i_data, :] = mu_i[:]

            cluster_id_matrix[i_data] = np.argmin(mu_matrix[i_data, :])

        # Decentralized clustering.
        own_cluster_id = cluster_id_matrix[0]

        cluster_members_indexes = \
            np.where(cluster_id_matrix == own_cluster_id)[0]
        if cluster_members_indexes.shape[0] > 1:
            cluster_center = local_free_preys[own_cluster_id]
            cluster_other_members = \
                local_free_predators[cluster_members_indexes[1:], :]
        else:
            cluster_center = local_free_preys[own_cluster_id]
            cluster_other_members = np.zeros((0, 2), dtype=int)

        return role, cluster_center, cluster_other_members, \
            self.captured_targets_map.copy(), self.locked_pursuers_map.copy()

    def is_captured(self, prey_position, local_env_matrix, offsets_from_local_positions_to_global_positions):

        capture_positions = self.get_valid_axial_neighbors(prey_position, self.fov_scope)
        local_full_capture_status = True if len(capture_positions) == 4 else False
        occupied_capture_positions = \
            local_env_matrix[capture_positions[:, 0],
                             capture_positions[:, 1], :].sum(axis=1)

        # Valid only if collision is not allowed in the space.
        # Otherwise, more than one agents can occupy the same position.
        yes_no = True if (occupied_capture_positions > 0).all() and len(capture_positions) == 4 else False

        global_prey_position = prey_position + offsets_from_local_positions_to_global_positions
        once_captured_globally = self.captured_targets_map[global_prey_position[0], global_prey_position[1]]
        if yes_no:
            if not once_captured_globally:
                self.captured_targets_map[global_prey_position[0], global_prey_position[1]] = 1
                global_locked_pursuers = \
                    self.get_valid_axial_neighbors(global_prey_position, self.locked_pursuers_map.shape[0])
                self.locked_pursuers_map[global_locked_pursuers[:, 0], global_locked_pursuers[:, 1]] = 1
        else:
            if not local_full_capture_status:
                # local uncaptured conclusion is drawn based on incomplete information,
                # then refer to the global record.
                if once_captured_globally:
                    yes_no = True
            elif once_captured_globally:
                # conflict between local and global due to dynamic env, update global since local is certain.
                self.captured_targets_map[global_prey_position[0], global_prey_position[1]] = 0

        return yes_no

    def get_valid_axial_neighbors(self, position, scope):
        axial_neighbors = position + self.axial_neighbors_mask

        valid_index = []
        for idx, position in enumerate(axial_neighbors):
            if (position >= 0).all() and (position < scope).all():
                valid_index.append(idx)

        axial_neighbors = axial_neighbors[valid_index, :]
        return axial_neighbors

