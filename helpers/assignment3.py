import osmnx
import numpy as np
import networkx as nx

from itertools import cycle
from matplotlib.colors import TABLEAU_COLORS


def get_network(center, radius):
    """ Get a networkx MultiDiGraph object representing the area specified in the query. """
    return osmnx.graph.graph_from_address(center, radius, network_type='walk', simplify=True)

    
def make_instance(graph):
    """ Create a single shortest path routing instance from a given graph.
        Returns a list of nodes and a dictionary of edges and their distances. """
    nodes = list(graph.nodes)
    edges = nx.get_edge_attributes(nx.DiGraph(graph), "length")
    
    pred = {i: [j for j in nodes if (j, i) in edges] for i in nodes}
    succ = {i: [j for j in nodes if (i, j) in edges] for i in nodes}

    return nodes, edges, pred, succ
    
    
def plot_network(graph, *routes):
    """ Plot a network instance, optionally including one or more routes. """
    cmap = cycle(TABLEAU_COLORS.keys())
    
    if len(routes) == 0:
        osmnx.plot_graph(graph)
    elif len(routes) == 1:
        osmnx.plot_graph_route(graph, routes[0])
    else:
        colors = [c for _, c in zip(routes, cmap)]
        osmnx.plot_graph_routes(graph, routes, colors)
