import math
import random
import rustworkx as rx
from rustworkx.visualization import graphviz_draw

"""
Edge-colors an undirected bipartite graph without multiple edges.

Follows the algorithm described in the paper 
"A simple algorithm for edge-coloring bipartite multigraphs" by Noga Alon, 2003.

The graphs are represented using rustworkx.
"""

####################################################################################
# auxiliary
####################################################################################

# Function to find the smallest power of 2
# greater than or equal to n
def nearestPowerOf2(N):
    a = int(math.log2(N))
    if 2**a == N:
        return N
    return 2 ** (a + 1)


####################################################################################
# checking whether a graph is bipartite
####################################################################################

def is_bipartite(graph: rx.PyGraph):
    """
    Checks if a given undirected graph is bipartite.
    Returns mapping from node to 0/1 if so.
    Returns None if not.
    """

    def visit(node, node_color):
        unprocessed = [(node, node_color)]
        while len(unprocessed) > 0:
            (n, c) = unprocessed.pop()
            if color[n] is not None and color[n] != c:
                return False
            color[n] = c
            if expanded[n]:
                continue
            for n1 in graph.neighbors(n):
                unprocessed.append((n1, 1 - c))
            expanded[n] = True
        return True

    num_nodes = len(graph.node_indices())
    expanded = [False] * num_nodes
    color = [None] * num_nodes

    for node in graph.node_indices():
        if color[node] is None:
            if not visit(node, 0):
                return None

    return color


####################################################################################
# finding euler path in a graph
####################################################################################

# def find_euler_path(graph, check_for_cycle=False):
#     graph = graph.copy()
#     nodes_with_odd_degree = []
#     for node in graph.node_indices():
#         if graph.degree(node) % 2 == 1:
#             nodes_with_odd_degree.append(node)
#     if check_for_cycle and len(nodes_with_odd_degree) > 0:
#         return None
#     if len(nodes_with_odd_degree) > 2:
#         return None
#     start_node = nodes_with_odd_degree[0] if len(nodes_with_odd_degree) > 0 else 0
#     stack = [start_node]
#     path = []
#     while len(stack) > 0:
#         n = stack[-1]
#         neighbors = graph.neighbors(n)
#         if len(neighbors) == 0:
#             path.append(n)
#             stack.pop()
#         else:
#             n1 = neighbors[0]
#             stack.append(n1)
#             graph.remove_edge(n, n1)
#     return path


def find_euler_cycles(graph):
    """
    Returns a set of Euler cycles for a possibly disconnected graph.

    An Euler cycle (or circuit) for a connected graph is a path that visits
    each edge of the graph exactly once and comes back to the starting point.

    A connected graph has an Euler cycle iff all of its vertices have even
    degree.
    """
    for node in graph.node_indices():
        if graph.degree(node) % 2 == 1:
            print(f"No euler cycles: node {node} has odd degree.")
            return None
    graph = graph.copy()
    processed = [False] * len(
        graph.node_indices()
    )  # nodes from already covered components

    all_cycles = []
    for node in graph.node_indices():
        if processed[node]:
            continue
        stack = [node]
        cycle = []
        while len(stack) > 0:
            n = stack[-1]
            neighbors = graph.neighbors(n)
            if len(neighbors) == 0:
                cycle.append(n)
                processed[n] = True
                stack.pop()
            else:
                n1 = neighbors[0]
                stack.append(n1)
                graph.remove_edge(n, n1)
        all_cycles.append(cycle)
    return all_cycles


####################################################################################
# Main algorithmics
####################################################################################

class EdgeData:
    def __init__(self, m, b=False):
        self.m = m  # multiplicity
        self.b = b  # bad

    def __repr__(self):
        res = str(self.m)
        if self.b:
            res += "x"
        return res


class BipartiteMultiGraph:
    def __init__(self, graph=None):
        self.graph = rx.PyGraph()
        self.left = []
        self.right = []
        self.degree = 0

        if graph is not None:
            color = is_bipartite(graph)
            if color is None:
                raise Exception("The graph is not bipartite")
            for node in graph.node_indices():
                self.graph.add_node(node)
            for edge in graph.edge_list():
                self.add_edge(edge[0], edge[1], 1)
            self.left = [v for v in graph.node_indices() if color[v] == 0]
            self.right = [v for v in graph.node_indices() if color[v] == 1]
            self.to_regular(graph)

    def set_nodes(self, node_indices, left_nodes, right_nodes):
        for node in node_indices:
            self.graph.add_node(node)
        self.left = left_nodes
        self.right = right_nodes

    def print(self):
        print(f"L-nodes: {self.left}")
        print(f"R-nodes: {self.right}")
        print(f"degree = {self.degree}")
        print(self.graph.edge_index_map())

    def get_degree(self, node_a):
        """Computes the total degree of node_a, accounting for the multiplicity of each edge."""
        degree = sum(
            self.get_m(node_a, node_b) for node_b in self.graph.neighbors(node_a)
        )
        return degree

    def get_num_bad_edges(self):
        """Total number of bad edges (ignoring multiplicity)."""
        num_bad_edges = 0
        for node_a in self.left:
            for node_b in self.graph.neighbors(node_a):
                if self.get_b(node_a, node_b):
                    num_bad_edges += 1
        return num_bad_edges

    def add_edge(self, node_a, node_b, m, bad=False):
        """
        Adds an edge (node_a, node_b) with multiplicity m if the edge
        does not already exist (or increases its multiplicity by m if it
        does).

        If the edge does not already exist, optionally marks it as bad
        based on the argument.
        """
        if m == 0:
            return
        if not self.graph.has_edge(node_a, node_b):
            self.graph.add_edge(node_a, node_b, EdgeData(m, bad))
        else:
            m0 = self.get_m(node_a, node_b)
            b0 = self.get_b(node_a, node_b)
            self.graph.update_edge(node_a, node_b, EdgeData(m0 + m, b0))

    def remove_edge(self, node_a, node_b):
        """
        Removes one instance of the edge (node_a, node_b).
        """
        m = self.get_m(node_a, node_b)
        b = self.get_b(node_a, node_b)
        if m == 1:
            self.graph.remove_edge(node_a, node_b)
        else:
            self.graph.update_edge(node_a, node_b, EdgeData(m - 1, b))

    def get_m(self, node_a, node_b):
        return self.graph.get_edge_data(node_a, node_b).m

    def get_b(self, node_a, node_b):
        return self.graph.get_edge_data(node_a, node_b).b

    def to_regular(self, graph):
        """Add extra nodes and edges to extend the graph to a regular multi-graph."""

        degree = max(graph.degree(v) for v in graph.node_indices())
        self.degree = degree
        # print(f"aiming degree = {degree}")

        # todo: add optimization to combine nodes on the same side if their total degree does not exceed k
        # add missing nodes
        num_left = len(self.left)
        num_right = len(self.right)
        if num_left > num_right:
            for _ in range(num_left - num_right):
                node = self.graph.add_node("")
                self.right.append(node)
        else:
            for _ in range(num_right - num_left):
                node = self.graph.add_node("")
                self.left.append(node)
        assert len(self.left) == len(self.right)
        n = len(self.left)

        # todo: this is linear, but we can optimize this a bit
        l_index = 0
        r_index = 0
        while l_index < n:
            l_degree = self.get_degree(self.left[l_index])
            if l_degree == degree:
                l_index += 1
                continue
            r_degree = self.get_degree(self.right[r_index])
            if r_degree == degree:
                r_index += 1
                continue
            m = min(degree - l_degree, degree - r_degree)
            self.add_edge(self.left[l_index], self.right[r_index], m)

    def add_matching(self, matching):
        """
        Adds edges corresponding to a perfect matching;
        increases the total degree by 1.
        """
        for (node_a, node_b) in matching:
            self.add_edge(node_a, node_b, 1)
        self.degree += 1

    def remove_matching(self, matching):
        """
        Removes edges corresponding to a perfect matching;
        decreases the total degree by 1.
        """
        for node_a, node_b in matching:
            self.remove_edge(node_a, node_b)
        self.degree -= 1


def split_regular_bipartite_multigraph_into_two(H0):
    """
    Given a regular bipartite multigraph H0 of an even degree 2r,
    split it into two regular bipartite multigraphs of degree r.
    """
    if H0.degree % 2 == 1:
        raise Exception(f"Cannot split regular graph of odd degree {H0.degree}")

    H1 = BipartiteMultiGraph()
    H2 = BipartiteMultiGraph()
    R = rx.PyGraph()

    H1.set_nodes(H0.graph.node_indices(), H0.left, H0.right)
    H1.degree = H0.degree // 2

    H2.set_nodes(H0.graph.node_indices(), H0.left, H0.right)
    H2.degree = H0.degree // 2

    for node in H0.graph.node_indices():
        R.add_node(node)

    for node_a in H0.left:
        for node_b in H0.graph.neighbors(node_a):
            m = H0.get_m(node_a, node_b)
            b = H0.get_b(node_a, node_b)
            H1.add_edge(node_a, node_b, m // 2, b)
            H2.add_edge(node_a, node_b, m // 2, b)
            if m % 2 == 1:
                R.add_edge(node_a, node_b, b)

    rem_cycles = find_euler_cycles(R)
    for cycle in rem_cycles:
        for i in range(len(cycle) - 1):
            node_a = cycle[i]
            node_b = cycle[i + 1]

            b = R.get_edge_data(node_a, node_b)

            if i % 2 == 0:
                H1.add_edge(node_a, node_b, 1, b)
            else:
                H2.add_edge(node_a, node_b, 1, b)

    return H1, H2


def find_perfect_matching_in_regular_bipartite_multigraph(H: BipartiteMultiGraph):
    """
    Finds a perfect matching in a regular bipartite multigraph of an arbitrary
    degree r.

    The idea is to add more edges to the multigraph to increase its degree to a power
    of 2 (however these might be "bad edges" not present in the original multigraph),
    and then to find a matching by recursively splitting the multigraph (and always
    selecting the multigraph with fewer bad edges) until we find a multigraph of degree
    1 without bad edges (it always exists).
    """
    k = H.degree
    n = len(H.left)  # n/2 in the paper
    m = H.degree * n
    t = nearestPowerOf2(m)  # 2 ** t in the paper
    t2 = int(math.log2(t))

    alpha = t // k
    beta = t - k * alpha

    # Choose an arbitrary matching M: left[i] -- right[i]
    H1 = BipartiteMultiGraph()
    H1.set_nodes(H.graph.node_indices(), H.left, H.right)
    H1.degree = t

    for node_a in H.left:
        for node_b in H.graph.neighbors(node_a):
            m = H.get_m(node_a, node_b)
            H1.add_edge(node_a, node_b, m * alpha)

    for i in range(n):
        node_a = H.left[i]
        node_b = H.right[i]
        H1.add_edge(node_a, node_b, beta, bad=True)

    # Recursively split until we find a matching
    for _ in range(t2):
        H2, H3 = split_regular_bipartite_multigraph_into_two(H1)
        bad2 = H2.get_num_bad_edges()
        bad3 = H3.get_num_bad_edges()
        if bad2 < bad3:
            H1 = H2
        else:
            H1 = H3
    assert H1.get_num_bad_edges() == 0
    matching = get_matching_from_degree_one(H1)
    return matching


def get_matching_from_degree_one(G: BipartiteMultiGraph):
    """
    Auxiliary function to extract a matching from a regular bipartite multigraph
    of degree 1.
    """
    matching = []
    for node_a in G.left:
        nbd = G.graph.neighbors(node_a)
        assert len(nbd) == 1
        node_b = nbd[0]
        matching.append((node_a, node_b))
    return matching


def edge_color_regular_bipartite_multigraph_when_power2(G: BipartiteMultiGraph):
    """
    Edge-colors a regular bipartite multigraph whose degree is a power of 2.

    This can be done very efficiently by recursively splitting the multigraph
    into pairs of multigraphs of half-degree.
    """
    coloring = []

    k = G.degree

    if k == 1:
        matching = get_matching_from_degree_one(G)
        coloring.append(matching)
        assert len(coloring) == G.degree
        return coloring

    H1, H2 = split_regular_bipartite_multigraph_into_two(G)
    coloring1 = edge_color_regular_bipartite_multigraph_when_power2(H1)
    coloring2 = edge_color_regular_bipartite_multigraph_when_power2(H2)
    coloring.extend(coloring1)
    coloring.extend(coloring2)
    assert len(coloring) == k
    return coloring


def edge_color_regular_bipartite_multigraph(G: BipartiteMultiGraph):
    """
    Edge-colors a regular bipartite multigraph of an arbitrary degree r.

    Returns a list of perfect matchings, each matching corresponding to
    a color.

    This function uses the following building blocks:

    * an algorithm to find a single perfect matching in a regular bipartite multigraph of
      an arbitrary degree
    * recursion
    * adding perfect matchings to a regular bipartite multigraph to increase its degree to
      a power of 2
    * fast edge-coloring algorithm for regular bipartite multigraphs whose degree
      is a power of 2

    This way to organize the building blocks guarantees that the full algorithm runs in
    O (m log m) time. A simpler (but less efficient) algorithm could be obtained by finding
    and removing one matching at a time.
    """
    coloring = []

    k = G.degree

    if k == 0:
        # dummy case with no edges
        return coloring

    if k == 1:
        matching = get_matching_from_degree_one(G)
        coloring.append(matching)
        assert len(coloring) == k
        return coloring

    perfect_matching = None

    # If the degree is odd, we find a perfect matching and remove it
    if k % 2 == 1:
        perfect_matching = find_perfect_matching_in_regular_bipartite_multigraph(G)
        assert perfect_matching is not None
        G.remove_matching(perfect_matching)

    assert G.degree % 2 == 0

    # Split graph into two regular bipartite multigraphs of half-degree
    H1, H2 = split_regular_bipartite_multigraph_into_two(G)

    # Recursively color H1
    H1_coloring = edge_color_regular_bipartite_multigraph(H1)

    # Transfer some matchings from H1 to H2 to make the degree of H2
    # to be a power of 2
    r = nearestPowerOf2(H2.degree)
    num_classes_to_add = r - H2.degree

    for i in range(num_classes_to_add):
        H2.add_matching(H1_coloring[i])
    assert H2.degree == r

    # Edge-color H2 by recursive splitting
    coloring = edge_color_regular_bipartite_multigraph_when_power2(H2)

    # Add remaining matchings from H1
    coloring.extend(H1_coloring[num_classes_to_add:])

    # Also add the perfect matchings (if the degree is odd)
    if perfect_matching is not None:
        coloring.append(perfect_matching)

    assert len(coloring) == k
    return coloring


####################################################################################
# MAIN FUNCTION: edge-coloring undirected bipartite graph without multiple edges
####################################################################################


def edge_color_bipartite_graph(graph: rx.PyGraph):
    """
    Edge-colors a bipartite graph.

    Raises an error if the graph is not bipartite.

    Steps:
    (1) Extend the graph to a regular bipartite multigraph of degree r,
        where r in the maximum degree of a node
    (2) Edge-color the multigraph
    (3) Extract edge-coloring of the original graph
    """
    G = BipartiteMultiGraph(graph)
    coloring = edge_color_regular_bipartite_multigraph(G)

    # The final output is a dict from edge index in the original graph to color
    graph_colors = dict()
    for edge_index in graph.edge_indices():
        graph.update_edge_by_index(edge_index, (edge_index, False))

    for c, matching in enumerate(coloring):
        for node_a, node_b in matching:
            if graph.has_edge(node_a, node_b):
                data = graph.get_edge_data(node_a, node_b)
                edge_index, is_colored = data
                if not is_colored:
                    graph_colors[edge_index] = c
                    graph.update_edge_by_index(edge_index, (edge_index, True))

    check_colors(graph, graph_colors)
    return graph_colors


##################################################################################
# VISUALISATION
##################################################################################


def edge_attr_fn(edge):
    attr_dict = {}
    if edge == 0:
        attr_dict["color"] = "orange"
    elif edge == 1:
        attr_dict["color"] = "blue"
    elif edge == 2:
        attr_dict["color"] = "green"
    else:
        attr_dict["color"] = "red"
    return attr_dict


def plot_colors(graph, colors):
    for edge_index in graph.edge_indices():
        graph.update_edge_by_index(edge_index, colors[edge_index])
    out = graphviz_draw(graph, edge_attr_fn=edge_attr_fn, method="neato")
    out.show()


##################################################################################
# EXPERIMENTS / TESTING
##################################################################################


def check_colors(graph, colors):
    if len(graph.edge_indices()) == 0:
        return
    for edge_index in graph.edge_indices():
        graph.update_edge_by_index(edge_index, colors[edge_index])
    for node in graph.node_indices():
        edge_colors = [
            graph.get_edge_data(node, node_b) for node_b in graph.neighbors(node)
        ]
        if len(edge_colors) != len(set(edge_colors)):
            raise Exception(f"Problem: invalid edge-coloring at node {node}")
    max_node_degree = max(graph.degree(node) for node in graph.node_indices())
    num_colors_used = max(colors.values()) + 1
    if num_colors_used > max_node_degree:
        raise Exception(f"Problem: too many colors are used.")
    print("Coloring is OK")


def random_bipartite_graph(n: int, m: int, p: float):
    """n vertices on the left; n vertices on the right; p - probability of an edge."""
    graph = rx.PyGraph()
    for i in range(n + m):
        graph.add_node(i)
    for i in range(n):
        for j in range(m):
            if random.random() < p:
                graph.add_edge(i, n + j, None)
    return graph


def experiment_heavy_hex(n):
    graph = rx.generators.heavy_hex_graph(n)
    colors = edge_color_bipartite_graph(graph)
    print(f"Coloring of the graph:")
    print(colors)
    # uncomment to plot
    # plot_colors(graph, colors)


def experiment_ring(n):
    """Even cycles only else it's not bipartite."""

    graph = rx.generators.cycle_graph(n)
    colors = edge_color_bipartite_graph(graph)
    print(f"Coloring of the graph:")
    print(colors)
    # # uncomment to plot
    # plot_colors(graph, colors)


def experiment_random(n, m, p):
    """Even cycles only else it's not bipartite."""

    graph = random_bipartite_graph(n, m, p)
    num_nodes = len(graph.node_indices())
    num_edges = len(graph.edge_indices())
    max_degree = max(graph.degree(node) for node in graph.node_indices())
    print(
        f"Generated graph has {num_nodes} nodes, {num_edges} edges, {max_degree} max degree."
    )
    colors = edge_color_bipartite_graph(graph)
    print(f"Coloring of the graph:")
    print(colors)
    # uncomment to plot
    # plot_colors(graph, colors)


def run_testing():
    """Testing for random bipartite graphs."""
    for seed in range(10):
        for n in [5, 10, 15, 20]:
            for m in [5, 10, 15, 20]:
                for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    print("=================")
                    print(f"{seed = }, {n = }, {m = }, {p = }")
                    random.seed(seed)
                    experiment_random(n, m, p)


if __name__ == "__main__":
    experiment_heavy_hex(25)
    # experiment_ring(100)
    # run_testing()
