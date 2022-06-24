"""
clustering.py
~~~~~~~~~~~~~

Author: Lijun SUN.
Date: FRI APR 25 2021.
"""
import copy
import numpy as np


class Clustering:
    def __init__(self, fov_scope):
        self.fov_scope = fov_scope

        self.fov_radius = int(0.5 * (self.fov_scope - 1))

        # Relative position in one agent's local view.
        # If fov_scope = 7, own_position = [3, 3].
        self.own_position = np.array([self.fov_radius] * 2)

        # Other parameters.
        # N, E, S, W.
        self.axial_neighbors_mask = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        pass

    def fuzzy_c_means(self, local_env_vectors, local_env_matrix):
        """
        :param local_env_vectors:
        :param local_env_matrix
        :return: a tuple of 2 elements, (cluster_center, cluster_other_members)
            where cluster_center is a 2d numpy array of shape (2,),
            cluster_other_members is a 2d numpy array of shape (m, 2).

        Reference:
        [1] Fuzzy c-means (FCM) clustering:
        J. C. Bezdek, “Fuzzy Mathematics in Pattern Classification,”
        Ph.D. thesis, Cornell Univ., Ithaca, NY, 1973.
        Bezdek, James C. Pattern recognition with fuzzy objective function
        algorithms. Springer Science & Business Media, 2013.
        """
        local_env_vectors = copy.deepcopy(local_env_vectors)

        # Set of data.
        local_predators = local_env_vectors["local_predators"]

        # Cluster centers.
        local_free_preys = local_env_vectors["local_preys"]

        # Remove already captured preys from the candidate cluster centers.
        valid_index = []
        for idx, prey in enumerate(local_free_preys):
            if not self.is_captured(prey, local_env_matrix):
                valid_index.append(idx)

        local_free_preys = local_free_preys[valid_index, :]

        # No local preys, the predator itself forms a single cluster.
        if local_free_preys.shape[0] == 0:
            cluster_center = self.own_position
            cluster_other_members = local_predators

            return cluster_center, cluster_other_members

        local_predators = np.vstack((self.own_position, local_predators))

        # No. of data.
        n_data = len(local_predators)

        # No. of clusters.
        n_clusters = len(local_free_preys)

        # Membership function.
        mu_matrix = np.zeros((n_data, n_clusters))

        # Cluster identification.
        cluster_id_matrix = np.zeros((n_data,), dtype=int)

        # Fuzzier,
        # the degree of fuzziness of the membership (clustering) matrix.
        m = 1.25

        for i_data in range(n_data):
            mu_i = np.zeros((n_clusters,))

            for j_cluster in range(n_clusters):
                # Euclidean distance.
                # mu_i[j_cluster] = \
                #     np.sum((predators[i_data] - preys[j_cluster]) ** 2) \
                #     ** (1/(1-m))

                fov_distance = np.linalg.norm(local_predators[i_data] - local_free_preys[j_cluster], ord=np.inf)
                if fov_distance > self.fov_radius:
                    mu_i[j_cluster] = 0
                else:
                    # Manhattan distance.
                    mu_i[j_cluster] = np.linalg.norm(
                        (local_predators[i_data] - local_free_preys[j_cluster]),
                        ord=1) ** (2 / (1 - m))

            mu_i_sum = np.sum(mu_i)
            mu_matrix[i_data, :] = mu_i[:] / mu_i_sum

            # for j_cluster in range(n_clusters):
            #     mu_matrix[i_data, j_cluster] = mu_i[j_cluster] / mu_i_sum

            cluster_id_matrix[i_data] = \
                np.random.choice(n_clusters, p=mu_matrix[i_data, :])

        # Decentralized clustering.
        own_cluster_id = cluster_id_matrix[0]

        cluster_members_indexes = \
            np.where(cluster_id_matrix == own_cluster_id)[0]
        if cluster_members_indexes.shape[0] > 1:
            cluster_center = local_free_preys[own_cluster_id]
            cluster_other_members = \
                local_predators[cluster_members_indexes[1:], :]
        else:
            cluster_center = local_free_preys[own_cluster_id]
            cluster_other_members = np.zeros((0, 2), dtype=int)

        # Debug info.
        print('- mu_matrix:\n', mu_matrix)
        print('- cluster_id_matrix:\n', cluster_id_matrix)
        # print('- local_free_preys:\n', local_free_preys)
        # print('- local_predators:\n', local_predators)
        # print("- Cluster center:\n", cluster_center)
        # print("- Cluster members:\n", cluster_other_members)

        return cluster_center, cluster_other_members

    def is_captured(self, prey_position, local_env_matrix):

        capture_positions = prey_position + self.axial_neighbors_mask

        valid_index = []
        for idx, position in enumerate(capture_positions):
            if (position >= 0).all() and (position < self.fov_scope).all():
                valid_index.append(idx)

        capture_positions = capture_positions[valid_index, :]

        occupied_capture_positions = \
            local_env_matrix[capture_positions[:, 0],
                             capture_positions[:, 1], :].sum()

        # Valid only if collision is not allowed in the space.
        # Otherwise, more than one agents can occupy the same position.
        yes_no = True
        if occupied_capture_positions != 4:
            yes_no = False

        return yes_no


def demo_fuzzy_clustering():
    # Set of data.
    local_predators = np.array([[4, 2]])
    # Cluster centers.
    local_preys = np.array([[3, 1], [0, 2]])

    local_env_vectors = {
        "local_preys": local_preys,
        "local_predators": local_predators,
        "local_obstacles": None
    }

    fov_scope = 5
    local_env_matrix = np.zeros((fov_scope, fov_scope, 3))
    for prey in local_preys:
        local_env_matrix[prey[0], prey[1], 0] = 1
    for predator in local_predators:
        local_env_matrix[predator[0], predator[1], 1] = 1

    clustering = Clustering(fov_scope=fov_scope)
    cluster_center, cluster_other_members = \
        clustering.fuzzy_c_means(local_env_vectors, local_env_matrix)


def main():
    demo_fuzzy_clustering()


if __name__ == "__main__":
    main()
