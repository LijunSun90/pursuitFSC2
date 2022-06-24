"""
Reference:
    [1] John Hopcroft and Robert Tarjan. 1973. Algorithm 447: efficient algorithms for graph manipulation. Commun.
        ACM 16, 6 (June 1973), 372–378. DOI:https://doi.org/10.1145/362248.362272
    [2] https://cppsecrets.com/users/5629115104105118971091101011031055657495564103109971051084699111109/Python-Connected-Components-in-Graph.php
    [3] https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/

Complexity [1]:
- Space: proportional to max(V, E).
- Time: proportional to max(V, E).
where V is the number of vertices and E is the number of edges of the graph.

Author: Lijun SUN.
Date: WED 8 SEP 2021.
"""
import numpy as np


class SimpleGraph:
    def __init__(self, n_vertices=None, adjacency_matrix=None):

        if n_vertices is not None:
            self.n_vertices = n_vertices
            self.adjacency_matrix = np.zeros((n_vertices, n_vertices), dtype=int)
            self.adjacency_list = [[] for _ in range(self.n_vertices)]
        elif adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix
            self.n_vertices = len(adjacency_matrix)
            # self.n_edges = np.sum(graph) / 2
            self.adjacency_list = self.get_adjacency_list()
        else:
            raise EOFError("Class Graph: Must give either n_vertices or graph")

        # components list
        self.all_connected_components = []
        self.vertex_component_labels = [None for _ in range(self.n_vertices)]
        self.visited = [False for _ in range(self.n_vertices)]

    def reset(self):
        self.all_connected_components = []
        self.visited = [False for _ in range(self.n_vertices)]

    def add_edge(self, vertex_1, vertex_2):
        self.adjacency_matrix[[vertex_1, vertex_2], [vertex_2, vertex_1]] = 1
        self.adjacency_list[vertex_1].append(vertex_2)
        self.adjacency_list[vertex_2].append(vertex_1)

    def get_adjacency_list(self):
        self.adjacency_list = [[] for _ in range(self.n_vertices)]

        for vertex in range(self.n_vertices):
            self.adjacency_list[vertex] = np.nonzero(self.adjacency_matrix[vertex])[0].tolist()

        return self.adjacency_list.copy()

    def depth_first_search(self, vertex, component, component_idx):
        # Marking node as visited.
        self.visited[vertex] = True

        # Append node in the component list.
        component.append(vertex)
        self.vertex_component_labels[vertex] = component_idx

        # Visit neighbors of the current node.
        for neighbor in self.adjacency_list[vertex]:
            if not self.visited[neighbor]:
                self.depth_first_search(neighbor, component, component_idx)

    def get_all_connected_components(self):
        component_idx = -1
        for vertex in range(self.n_vertices):
            if not self.visited[vertex]:
                component_idx += 1
                component = []
                self.depth_first_search(vertex, component, component_idx)
                self.all_connected_components.append(component)

        return self.all_connected_components.copy(), self.vertex_component_labels.copy()

    def get_all_vertices_degrees(self):
        all_vertices_degrees = np.sum(self.adjacency_matrix, axis=1)

        return all_vertices_degrees


def test():
    adjacency_matrix = [
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]]
    # array([[0, 1, 1, 0, 0],
    #        [1, 0, 1, 0, 0],
    #        [1, 1, 0, 0, 0],
    #        [0, 0, 0, 0, 1],
    #        [0, 0, 0, 1, 0]])
    adjacency_matrix = np.transpose(adjacency_matrix) | adjacency_matrix

    graph_object = SimpleGraph(adjacency_matrix=adjacency_matrix)
    all_components, vertex_component_labels = graph_object.get_all_connected_components()

    # adjacency_list:
    #  [[1, 2], [0, 2], [0, 1], [4], [3]]
    # all_components:
    #  [[0, 1, 2], [3, 4]]
    # vertex_component_labels:
    #  [0, 0, 0, 1, 1]
    print("adjacency_list:\n", graph_object.adjacency_list)
    print("all_components:\n", all_components)
    print("vertex_component_labels:\n", vertex_component_labels)
    pass


if __name__ == "__main__":
    test()
