import math
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
    

def get_route(model):
    route = [model.s()]
    while route[-1] != model.t():
        route.append(max(model.nodes, key=lambda i: model.x[route[-1], i]() if (route[-1], i) in model.edges else -1))
    return route


def get_distances_to(graph, node):
    return nx.single_source_dijkstra_path_length(graph, node, weight="length")
 
 
def get_crowdedness(graph):
    c = 5395065019
    delta = get_distances_to(graph, c)
    return {k: 2*math.exp(-v/250) if k != c else 2 for k, v in delta.items()}
 
    
def plot_network(graph, *routes):
    """ Plot a network instance, optionally including one or more routes. """    
    if len(routes) == 0:
        osmnx.plot_graph(graph)
    elif len(routes) == 1:
        osmnx.plot_graph_route(graph, routes[0])
    else:
        cmap = cycle(TABLEAU_COLORS.keys())
        colors = [c for _, c in zip(routes, cmap)]
        osmnx.plot_graph_routes(graph, routes, colors)


def plot_network_heatmap(graph, route=None, node_color=None, edge_color=None, route_color=None):
    """ Plot a network instance, optionally including one or more routes. """
    if route is None:
        osmnx.plot_graph(graph, node_color=node_color, edge_color=edge_color)
    else:
        osmnx.plot_graph_route(graph, route, node_color=node_color, edge_color=edge_color, route_color=route_color)
 