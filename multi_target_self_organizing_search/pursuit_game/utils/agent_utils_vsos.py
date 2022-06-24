import numpy as np

from .discrete_agent import DiscreteAgent
from .simple_graph_vsos import SimpleGraph

#################################################################
# Implements utility functions for multi-agent DRL
#################################################################


def create_agents_vsop(n_pursuers, n_evaders, map_matrix, obs_range, randomizer,
                       flatten=False, randinit=False, constraints=None):
    """
    Initializes the agents on a map (map_matrix)
    -n_pursuers: the number of pursuers to put on the map
    -n_evaders: the number of evaders to put on the map
    -randinit: if True will place agents in random, feasible locations
               if False will place all agents at 0
    expanded_mat: This matrix is used to spawn non-adjacent agents
    """
    xs, ys = map_matrix.shape
    pursuers = []
    evaders = []
    expanded_mat = np.zeros((xs + 2, ys + 2))

    for i in range(n_pursuers):
        xinit, yinit = (0, 0)
        if randinit:
            xinit, yinit = feasible_position_exp(randomizer, map_matrix, expanded_mat, constraints=constraints)

        # fill expanded_mat
        expanded_mat[xinit + 1, yinit + 1] = -1
        expanded_mat[xinit + 2, yinit + 1] = -1
        expanded_mat[xinit, yinit + 1] = -1
        expanded_mat[xinit + 1, yinit + 2] = -1
        expanded_mat[xinit + 1, yinit] = -1
        agent = DiscreteAgent(xs, ys, map_matrix, randomizer,
                              obs_range=obs_range, flatten=flatten)
        agent.set_position(xinit, yinit)
        pursuers.append(agent)

    for i in range(n_evaders):
        xinit, yinit = (0, 0)
        if randinit:
            xinit, yinit = feasible_position_exp(randomizer, map_matrix, expanded_mat, constraints=constraints)

        # fill expanded_mat
        expanded_mat[xinit + 1, yinit + 1] = -1
        expanded_mat[xinit + 2, yinit + 1] = -1
        expanded_mat[xinit, yinit + 1] = -1
        expanded_mat[xinit + 1, yinit + 2] = -1
        expanded_mat[xinit + 1, yinit] = -1
        agent = DiscreteAgent(xs, ys, map_matrix, randomizer,
                              obs_range=obs_range, flatten=flatten)
        agent.set_position(xinit, yinit)
        evaders.append(agent)

    return pursuers, evaders


def feasible_position_exp(randomizer, map_matrix, expanded_mat, constraints=None):
    """
    Returns a feasible position on map (map_matrix)
    """
    xs, ys = map_matrix.shape
    while True:
        if constraints is None:
            x = randomizer.randint(0, xs)
            y = randomizer.randint(0, ys)
        else:
            xl, xu = constraints[0]
            yl, yu = constraints[1]
            x = randomizer.randint(xl, xu)
            y = randomizer.randint(yl, yu)
        if map_matrix[x, y] != -1 and expanded_mat[x + 1, y + 1] != -1:
            return (x, y)


def set_agents(agent_matrix, map_matrix):
    # check input sizes
    if agent_matrix.shape != map_matrix.shape:
        raise ValueError(
            "Agent configuration and map matrix have mis-matched sizes")

    agents = []
    xs, ys = agent_matrix.shape
    for i in range(xs):
        for j in range(ys):
            n_agents = agent_matrix[i, j]
            if n_agents > 0:
                if map_matrix[i, j] == -1:
                    raise ValueError(
                        "Trying to place an agent into a building: check map matrix and agent configuration")
                agent = DiscreteAgent(xs, ys, map_matrix)
                agent.set_position(i, j)
                agents.append(agent)
    return agents


def get_coordination_reward(pursuers_positions, evaders_positions, obs_radius, pursuers_complete_capture):
    # ##################################################

    n_pursuers = len(pursuers_positions)
    n_evaders = len(evaders_positions)

    # ##################################################
    # Build the graphs.
    pursuer_pursuer_graph = SimpleGraph(n_vertices=n_pursuers)
    pursuer_evader_graph = SimpleGraph(n_vertices=(n_pursuers + n_evaders))

    # Construct pursuer_pursuer_graph.
    for pursuer_1 in range(n_pursuers):
        for pursuer_2 in range(pursuer_1 + 1, n_pursuers):
            vector = pursuers_positions[pursuer_1] - pursuers_positions[pursuer_2]
            distance = np.linalg.norm(vector, ord=np.inf)
            if distance <= obs_radius:
                pursuer_pursuer_graph.add_edge(pursuer_1, pursuer_2)

    # Construct pursuer_evader_graph.
    for pursuer in range(n_pursuers):
        for evader in range(n_evaders):
            vector = pursuers_positions[pursuer] - evaders_positions[evader]
            distance = np.linalg.norm(vector, ord=np.inf)
            if distance <= obs_radius:
                pursuer_evader_graph.add_edge(pursuer, n_pursuers + evader)

    # ##################################################
    # Share the reward of finding evaders to all component pursuer members.
    # The reward amount is determined by the number of distinct evaders in the flock's perception.

    pursuer_evader_adjacency_list = pursuer_evader_graph.adjacency_list
    all_pursuer_pursuer_components, vertex_component_labels = pursuer_pursuer_graph.get_all_connected_components()

    find_n_evaders_for_component_pursuer = np.zeros(n_pursuers, dtype=int)

    for pursuer_pursuer_component in all_pursuer_pursuer_components:
        distinct_neighboring_evaders = set()
        for pursuer in pursuer_pursuer_component:
            current_neighboring_evaders = set(pursuer_evader_adjacency_list[pursuer])
            distinct_neighboring_evaders = distinct_neighboring_evaders.union(current_neighboring_evaders)

        find_n_evaders_for_component_pursuer[pursuer_pursuer_component] += len(distinct_neighboring_evaders)

    # ##################################################
    # Share the reward of capturing evaders to all component pursuer members.

    capture_evaders_for_component_pursuer = np.zeros(n_pursuers, dtype=np.bool)

    for pursuer, complete_capture in enumerate(pursuers_complete_capture):
        if complete_capture:
            component_label = vertex_component_labels[pursuer]
            component_members = all_pursuer_pursuer_components[component_label]
            capture_evaders_for_component_pursuer[component_members] = True

    # ##################################################
    # Return.
    return find_n_evaders_for_component_pursuer, capture_evaders_for_component_pursuer


def get_coordination_reward_v2(pursuers_positions, evaders_positions, obs_radius):
    # ##################################################

    n_pursuers = len(pursuers_positions)
    n_evaders = len(evaders_positions)

    # ##################################################
    # Build the graphs.
    pursuer_pursuer_graph = SimpleGraph(n_vertices=n_pursuers)
    pursuer_evader_capture_graph = SimpleGraph(n_vertices=(n_pursuers + n_evaders))

    for pursuer_1 in range(n_pursuers):
        # Construct pursuer_pursuer graph.
        for pursuer_2 in range(pursuer_1 + 1, n_pursuers):
            vector = pursuers_positions[pursuer_1] - pursuers_positions[pursuer_2]
            distance = np.linalg.norm(vector, ord=np.inf)
            if distance <= obs_radius:
                pursuer_pursuer_graph.add_edge(pursuer_1, pursuer_2)

        # Construct pursuer_evader_capture graph.
        pursuer_position_list = pursuers_positions[pursuer_1].tolist()
        for evader in range(n_evaders):
            if pursuer_position_list == evaders_positions[evader].tolist():
                pursuer_evader_capture_graph.add_edge(pursuer_1, n_pursuers + evader)

    # ##################################################
    # Share the reward of capturing n evaders to all component pursuer members.
    # The reward amount is determined by the number of distinct evaders the flock captures.

    all_pursuer_pursuer_components, vertex_component_labels = pursuer_pursuer_graph.get_all_connected_components()
    pursuer_evader_adjacency_list = pursuer_evader_capture_graph.adjacency_list

    capture_n_evaders_for_component_pursuer = np.zeros(n_pursuers, dtype=int)

    for pursuer_pursuer_component in all_pursuer_pursuer_components:
        distinct_captured_evaders = set()
        for pursuer in pursuer_pursuer_component:
            current_captured_evaders = set(pursuer_evader_adjacency_list[pursuer])
            distinct_captured_evaders = distinct_captured_evaders.union(current_captured_evaders)

        capture_n_evaders_for_component_pursuer[pursuer_pursuer_component] += len(distinct_captured_evaders)

    return capture_n_evaders_for_component_pursuer


def get_coordination_reward_v3(pursuers_positions, evaders_positions, obs_radius):
    # ##################################################

    n_pursuers = len(pursuers_positions)
    n_evaders = len(evaders_positions)

    # ##################################################
    # Build the graphs.
    pursuer_pursuer_graph = SimpleGraph(n_vertices=n_pursuers)
    pursuer_evader_capture_graph = SimpleGraph(n_vertices=(n_pursuers + n_evaders))

    for pursuer_1 in range(n_pursuers):
        # Construct pursuer_pursuer graph.
        for pursuer_2 in range(pursuer_1 + 1, n_pursuers):
            vector = pursuers_positions[pursuer_1] - pursuers_positions[pursuer_2]
            distance = np.linalg.norm(vector, ord=np.inf)
            if distance <= obs_radius:
                pursuer_pursuer_graph.add_edge(pursuer_1, pursuer_2)

        # Construct pursuer_evader_capture graph.
        pursuer_position_list = pursuers_positions[pursuer_1].tolist()
        for evader in range(n_evaders):
            if pursuer_position_list == evaders_positions[evader].tolist():
                pursuer_evader_capture_graph.add_edge(pursuer_1, n_pursuers + evader)

    # ##################################################
    # Share the reward of capturing n evaders to all component pursuer members.
    # The reward amount is determined by the number of distinct evaders the flock captures.

    all_pursuer_pursuer_components, vertex_component_labels = pursuer_pursuer_graph.get_all_connected_components()
    pursuer_evader_adjacency_list = pursuer_evader_capture_graph.adjacency_list

    capture_n_evaders_for_component_pursuer = np.zeros(n_pursuers, dtype=int)

    for pursuer_pursuer_component in all_pursuer_pursuer_components:
        distinct_captured_evaders = set()
        for pursuer in pursuer_pursuer_component:
            current_captured_evaders = set(pursuer_evader_adjacency_list[pursuer])
            distinct_captured_evaders = distinct_captured_evaders.union(current_captured_evaders)

        capture_n_evaders_for_component_pursuer[pursuer_pursuer_component] += len(distinct_captured_evaders)

    # ##################################################
    # Get all vertices (pursuers) degrees as an indicator of its neighborhood density.

    all_pursuers_degrees = pursuer_pursuer_graph.get_all_vertices_degrees()

    return capture_n_evaders_for_component_pursuer, all_pursuers_degrees


def get_coordination_reward_v4(pursuers_positions, evaders_positions, obs_radius):
    # ##################################################

    n_pursuers = len(pursuers_positions)
    n_evaders = len(evaders_positions)

    # ##################################################
    # Build the graphs.
    pursuer_pursuer_graph = SimpleGraph(n_vertices=n_pursuers)
    pursuer_evader_capture_graph = SimpleGraph(n_vertices=(n_pursuers + n_evaders))

    for pursuer_1 in range(n_pursuers):
        # Construct pursuer_pursuer graph.
        for pursuer_2 in range(pursuer_1 + 1, n_pursuers):
            vector = pursuers_positions[pursuer_1] - pursuers_positions[pursuer_2]
            distance = np.linalg.norm(vector, ord=np.inf)
            if distance <= obs_radius:
                pursuer_pursuer_graph.add_edge(pursuer_1, pursuer_2)

        # Construct pursuer_evader_capture graph.
        pursuer_position_list = pursuers_positions[pursuer_1].tolist()
        for evader in range(n_evaders):
            if pursuer_position_list == evaders_positions[evader].tolist():
                pursuer_evader_capture_graph.add_edge(pursuer_1, n_pursuers + evader)

    # ##################################################
    # Get all vertices (pursuers) degrees as an indicator of its neighborhood density.

    all_pursuers_degrees = pursuer_pursuer_graph.get_all_vertices_degrees()

    # ##################################################
    # Scaled shared reward of capturing evaders to component pursuer members.

    all_pursuer_pursuer_components, vertex_component_labels = pursuer_pursuer_graph.get_all_connected_components()
    pursuer_evader_adjacency_list = pursuer_evader_capture_graph.adjacency_list

    scaled_shared_reward = np.zeros(n_pursuers, dtype=int)

    for pursuer_pursuer_component in all_pursuer_pursuer_components:
        distinct_captured_evaders = set()
        for pursuer in pursuer_pursuer_component:
            current_captured_evaders = set(pursuer_evader_adjacency_list[pursuer])
            distinct_captured_evaders = distinct_captured_evaders.union(current_captured_evaders)
            scaled_shared_reward[pursuer] += len(current_captured_evaders)

        for pursuer in pursuer_pursuer_component:
            scaled_shared_reward[pursuer] += len(distinct_captured_evaders) * np.exp(-all_pursuers_degrees[pursuer])

    return scaled_shared_reward


def get_coordination_reward_v5(pursuers_positions, evaders_positions, obs_radius):
    # ##################################################

    n_pursuers = len(pursuers_positions)
    n_evaders = len(evaders_positions)

    # ##################################################
    # Build the graphs.
    pursuer_pursuer_graph = SimpleGraph(n_vertices=n_pursuers)
    pursuer_evader_capture_graph = SimpleGraph(n_vertices=(n_pursuers + n_evaders))

    for pursuer_1 in range(n_pursuers):
        # Construct pursuer_pursuer graph.
        for pursuer_2 in range(pursuer_1 + 1, n_pursuers):
            vector = pursuers_positions[pursuer_1] - pursuers_positions[pursuer_2]
            distance = np.linalg.norm(vector, ord=np.inf)
            if distance <= obs_radius:
                pursuer_pursuer_graph.add_edge(pursuer_1, pursuer_2)

        # Construct pursuer_evader_capture graph.
        pursuer_position_list = pursuers_positions[pursuer_1].tolist()
        for evader in range(n_evaders):
            if pursuer_position_list == evaders_positions[evader].tolist():
                pursuer_evader_capture_graph.add_edge(pursuer_1, n_pursuers + evader)

    # ##################################################
    # Get all vertices (pursuers) degrees as an indicator of its neighborhood density.

    all_pursuers_degrees = pursuer_pursuer_graph.get_all_vertices_degrees()

    # ##################################################
    # Scaled shared reward of capturing evaders to component pursuer members.

    all_pursuer_pursuer_components, vertex_component_labels = pursuer_pursuer_graph.get_all_connected_components()
    pursuer_evader_adjacency_list = pursuer_evader_capture_graph.adjacency_list

    scaled_shared_reward = np.zeros(n_pursuers, dtype=int)

    n_components = len(all_pursuer_pursuer_components)
    capture_evaders_for_component = np.zeros(n_components, dtype=np.bool)

    for idx, pursuer_pursuer_component in enumerate(all_pursuer_pursuer_components):
        for pursuer in pursuer_pursuer_component:
            current_captured_evaders = set(pursuer_evader_adjacency_list[pursuer])
            if len(current_captured_evaders) > 0:
                capture_evaders_for_component[idx] = True
                scaled_shared_reward[pursuer] = 1

        scaled_shared_reward[pursuer_pursuer_component] = scaled_shared_reward[pursuer_pursuer_component] + \
            capture_evaders_for_component[idx] * np.exp(-all_pursuers_degrees[pursuer_pursuer_component])

    return scaled_shared_reward


def get_coordination_reward_v6(pursuers_positions, evaders_positions, obs_radius):
    # ##################################################

    n_pursuers = len(pursuers_positions)
    n_evaders = len(evaders_positions)

    # ##################################################
    # Build the graphs.
    pursuer_pursuer_graph = SimpleGraph(n_vertices=n_pursuers)
    pursuer_evader_capture_graph = SimpleGraph(n_vertices=(n_pursuers + n_evaders))

    for pursuer_1 in range(n_pursuers):
        # Construct pursuer_pursuer graph.
        for pursuer_2 in range(pursuer_1 + 1, n_pursuers):
            vector = pursuers_positions[pursuer_1] - pursuers_positions[pursuer_2]
            distance = np.linalg.norm(vector, ord=np.inf)
            if distance <= obs_radius:
                pursuer_pursuer_graph.add_edge(pursuer_1, pursuer_2)

        # Construct pursuer_evader_capture graph.
        pursuer_position_list = pursuers_positions[pursuer_1].tolist()
        for evader in range(n_evaders):
            if pursuer_position_list == evaders_positions[evader].tolist():
                pursuer_evader_capture_graph.add_edge(pursuer_1, n_pursuers + evader)

    # ##################################################
    # Shared reward of capturing evaders to component pursuer members.

    all_pursuer_pursuer_components, vertex_component_labels = pursuer_pursuer_graph.get_all_connected_components()
    pursuer_evader_adjacency_list = pursuer_evader_capture_graph.adjacency_list

    shared_reward = np.zeros(n_pursuers, dtype=int)

    n_components = len(all_pursuer_pursuer_components)
    capture_evaders_for_component = np.zeros(n_components, dtype=np.bool)

    for idx, pursuer_pursuer_component in enumerate(all_pursuer_pursuer_components):
        for pursuer in pursuer_pursuer_component:
            current_captured_evaders = set(pursuer_evader_adjacency_list[pursuer])
            if len(current_captured_evaders) > 0:
                capture_evaders_for_component[idx] = True
                break

        shared_reward[pursuer_pursuer_component] = capture_evaders_for_component[idx]

    return shared_reward

