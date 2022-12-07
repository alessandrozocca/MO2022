import osmnx
import numpy as np
import networkx as nx

from itertools import cycle
from matplotlib.colors import TABLEAU_COLORS


cmap = cycle(TABLEAU_COLORS.keys())


def get_network(query):
    """ Get a networkx MultiDiGraph object representing the area specified in the query. """
    return osmnx.graph.graph_from_place(query, network_type='drive', simplify=True)

    
def make_instance(graph, q=0.75):
    """ Create a single shortest path routing instance from a given graph.
        Returns source s, target t, a list of nodes, and a dictionary of edges and their distances. """
    nodes = list(graph.nodes)
    edges = nx.get_edge_attributes(nx.DiGraph(graph), "length")
    
    distance_matrix = nx.floyd_warshall_numpy(graph, weight="length")
    quantile = np.quantile(distance_matrix, q)

    s = nodes[np.abs(distance_matrix - quantile).min(1).argmin()]
    t = nodes[np.abs(distance_matrix - quantile).min(0).argmin()]

    return s, t, nodes, edges
    
    
def plot_network(graph, *routes):
    """ Plot a network instance, optionally including one or more routes. """
    if len(routes) == 0:
        osmnx.plot_graph(graph)
    elif len(routes) == 1:
        osmnx.plot_graph_route(graph, routes[0])
    else:
        colors = [c for _, c in zip(routes, cmap)]
        osmnx.plot_graph_routes(graph, routes, colors)
