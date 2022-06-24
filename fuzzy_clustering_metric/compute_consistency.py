import numpy as np
import itertools


def compute_stochastic_consistency(fuzzy_membership_matrix_list):
    # Assume each agent's fuzzy membership matrix includes all agents and all clusters.
    # Row: agent_i; Column: cluster_j; Value: fuzzy membership value of agent i to cluster j.
    n_agents, n_targets = fuzzy_membership_matrix_list[0].shape

    all_clusters_possibilities = []
    for i_row in range(n_agents):
        fuzzy_membership_matrix_agent_i = fuzzy_membership_matrix_list[i_row]
        for j_column in range(n_agents):
            valid_clusters_of_agent_j = np.nonzero(fuzzy_membership_matrix_agent_i[j_column, :])[0].tolist()
            n_valid = len(valid_clusters_of_agent_j)
            if n_valid == 0:
                # -1 means agent i has no idea of the possible cluster of agent j.
                all_clusters_possibilities.append([-1])
            else:
                all_clusters_possibilities.append(valid_clusters_of_agent_j)

    all_clusters_possibilities = list(itertools.product(*all_clusters_possibilities))

    # Fill the consistency matrix row-by-row.
    # Row i: the cluster allocations to all agents by agent i.
    # Row i, Column j: the cluster allocation for agent j by agent i, where -1 means agent i has no idea.

    stochastic_consistency = 0
    for one_possibility in all_clusters_possibilities:
        consistency_idx_matrix = np.reshape(one_possibility, (n_agents, n_agents))
        consistency_matrix = np.ones((n_agents, n_agents)) * -1

        total_probability = 1
        for i_row in range(n_agents):
            for j_column in range(n_agents):
                cluster_idx = consistency_idx_matrix[i_row, j_column]
                if cluster_idx == -1:
                    probability = 1
                else:
                    consistency_matrix[i_row, j_column] = cluster_idx
                    probability = fuzzy_membership_matrix_list[i_row][j_column][cluster_idx]

                total_probability *= probability

        partial_consistency = compute_consistency(consistency_matrix)

        stochastic_consistency += total_probability * partial_consistency

    return stochastic_consistency


def compute_consistency(consistency_matrix):
    n_rows, n_columns = consistency_matrix.shape

    consistency = 0
    for i_row in range(n_rows - 1):
        i_row_valid_columns = np.where(consistency_matrix[i_row, :] != -1)[0]
        for j_row in range(i_row + 1, n_rows):
            j_row_valid_columns = np.where(consistency_matrix[j_row, :] != -1)[0]

            common_valid_columns = list(set(i_row_valid_columns).intersection(j_row_valid_columns))
            n_common_valid_columns = len(common_valid_columns)
            if n_common_valid_columns == 0:
                consistency += 1
                continue

            n_common = \
                consistency_matrix[i_row, common_valid_columns] - consistency_matrix[j_row, common_valid_columns]
            n_common = n_common.tolist().count(0)
            consistency += n_common / n_common_valid_columns

    consistency /= n_rows * (n_rows - 1) / 2

    return consistency


def test_consistency():
    membership_matrix = np.array([[1, 1], [1, 1]])
    consistency = compute_consistency(membership_matrix)
    print("consistency", consistency)
    membership_matrix = np.array([[1, 1], [1, 2]])
    consistency = compute_consistency(membership_matrix)
    print("consistency", consistency)
    membership_matrix = np.array([[1, -1], [-1, 2]])
    consistency = compute_consistency(membership_matrix)
    print("consistency", consistency)


def test_stochastic_consistency():
    # fuzzy_membership_matrix_agent_1 = np.array([[0.06, 0.94],
    #                                             [0.06, 0.94]])
    # fuzzy_membership_matrix_agent_2 = np.array([[0.06, 0.94],
    #                                             [0.06, 0.94]])
    # fuzzy_membership_matrix_list = [fuzzy_membership_matrix_agent_1, fuzzy_membership_matrix_agent_2]
    # stochastic_consistency = compute_stochastic_consistency(fuzzy_membership_matrix_list)
    # # 0.887
    # print('stochastic_consistency:', stochastic_consistency)

    # fuzzy_membership_matrix_agent_1 = np.array([[1, 0],
    #                                             [1, 0]])
    # fuzzy_membership_matrix_agent_2 = np.array([[0.835, 0.165],
    #                                             [0.06, 0.94]])
    # fuzzy_membership_matrix_list = [fuzzy_membership_matrix_agent_1, fuzzy_membership_matrix_agent_2]
    # stochastic_consistency = compute_stochastic_consistency(fuzzy_membership_matrix_list)
    # # 0.4475
    # print('stochastic_consistency:', stochastic_consistency)

    # fuzzy_membership_matrix_agent_1 = np.array([[1, 0],
    #                                             [1, 0]])
    # fuzzy_membership_matrix_agent_2 = np.array([[1, 0],
    #                                             [0.06, 0.94]])
    # fuzzy_membership_matrix_list = [fuzzy_membership_matrix_agent_1, fuzzy_membership_matrix_agent_2]
    # stochastic_consistency = compute_stochastic_consistency(fuzzy_membership_matrix_list)
    # # 0.53
    # print('stochastic_consistency:', stochastic_consistency)

    # fuzzy_membership_matrix_agent_1 = np.array([[1, 0],
    #                                             [1, 0]])
    # fuzzy_membership_matrix_agent_2 = np.array([[1, 0],
    #                                             [0.004, 0.996]])
    # fuzzy_membership_matrix_list = [fuzzy_membership_matrix_agent_1, fuzzy_membership_matrix_agent_2]
    # stochastic_consistency = compute_stochastic_consistency(fuzzy_membership_matrix_list)
    # # 0.502
    # print('stochastic_consistency:', stochastic_consistency)

    # fuzzy_membership_matrix_agent_1 = np.array([[0.5, 0.5, 0],
    #                                             [0.5, 0.5, 0]])
    # fuzzy_membership_matrix_agent_2 = np.array([[0.5, 0.5, 0],
    #                                             [0.055, 0.055, 0.89]])
    # fuzzy_membership_matrix_list = [fuzzy_membership_matrix_agent_1, fuzzy_membership_matrix_agent_2]
    # stochastic_consistency = compute_stochastic_consistency(fuzzy_membership_matrix_list)
    # # 0.2775
    # print('stochastic_consistency:', stochastic_consistency)

    fuzzy_membership_matrix_agent_1 = np.array([[0.5, 0.5, 0],
                                                [0.5, 0.5, 0]])
    fuzzy_membership_matrix_agent_2 = np.array([[0.5, 0.5, 0],
                                                [0, 0, 1]])
    fuzzy_membership_matrix_list = [fuzzy_membership_matrix_agent_1, fuzzy_membership_matrix_agent_2]
    stochastic_consistency = compute_stochastic_consistency(fuzzy_membership_matrix_list)
    # 0.25
    print('stochastic_consistency:', stochastic_consistency)

    # fuzzy_membership_matrix_agent_1 = np.array([[0.06, 0.94, 0],
    #                                             [0.06, 0.94, 0]])
    # fuzzy_membership_matrix_agent_2 = np.array([[0.06, 0.94, 0],
    #                                             [0.055, 0.89, 0.055]])
    # fuzzy_membership_matrix_list = [fuzzy_membership_matrix_agent_1, fuzzy_membership_matrix_agent_2]
    # stochastic_consistency = compute_stochastic_consistency(fuzzy_membership_matrix_list)
    # # 0.8635
    # print('stochastic_consistency:', stochastic_consistency)

    # fuzzy_membership_matrix_agent_1 = np.array([[0.06, 0.94, 0],
    #                                             [0.06, 0.94, 0]])
    # fuzzy_membership_matrix_agent_2 = np.array([[0.06, 0.94, 0],
    #                                             [0.03, 0.485, 0.485]])
    # fuzzy_membership_matrix_list = [fuzzy_membership_matrix_agent_1, fuzzy_membership_matrix_agent_2]
    # stochastic_consistency = compute_stochastic_consistency(fuzzy_membership_matrix_list)
    # # 0.6725
    # print('stochastic_consistency:', stochastic_consistency)
    pass


if __name__ == "__main__":
    # test_consistency()
    test_stochastic_consistency()
    pass
