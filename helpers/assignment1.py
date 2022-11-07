import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import perf_counter


def generate_instance(n_nodes, seed):
    """ Generate a random instance of size [n_nodes] with a fixed [seed] """
    rng = np.random.default_rng(seed)

    # Hardcoded instance parameters
    ARRIVALS_SCALE = 5 * n_nodes
    DEMAND_MIN, DEMAND_MAX = -15, 15
    COORD_MIN, COORD_MAX = 0, 100
    CENTER_MEAN_QBOX = .25, .75
    CENTER_STD_QBOX = .1, .15
    MIN_AVG_CLUSTER_SIZE = 5

    # Draw time window opening times exponentially
    arrival_times = rng.exponential(ARRIVALS_SCALE, n_nodes - 1)
    earliest_arrivals = np.append(0, arrival_times).astype(int)
    
    # Draw demands uniformly
    demands = rng.integers(DEMAND_MIN, DEMAND_MAX, n_nodes - 2)
    
    # Make sure the demands sum up to zero
    demand_left = np.sum(demands)
    demands = np.append(max(-demand_left, 0), demands)
    demands = np.append(demands, min(-demand_left, 0))

    # Assign each node to a cluster
    n_clusters = rng.integers(1, n_nodes // MIN_AVG_CLUSTER_SIZE + 1)
    clusters = rng.integers(0, n_clusters, n_nodes)
    
    # Compute the coordinate range
    coord_range = COORD_MAX - COORD_MIN

    # Draw the cluster means unformly between given limits
    mean_qbox = np.multiply(CENTER_MEAN_QBOX, coord_range) + COORD_MIN

    x_means = rng.uniform(*mean_qbox, n_clusters)
    y_means = rng.uniform(*mean_qbox, n_clusters)
    
    # Draw the cluster stds unformly between given limits
    std_qbox = np.multiply(CENTER_STD_QBOX, coord_range)

    x_stds = rng.uniform(*std_qbox, n_clusters)
    y_stds = rng.uniform(*std_qbox, n_clusters)

    # Create uniformly distributed clusters (indepentently for x and y)
    x_coords = x_means[clusters] + x_stds[clusters] * rng.standard_normal(n_nodes)
    y_coords = y_means[clusters] + y_stds[clusters] * rng.standard_normal(n_nodes)

    # Bound and round each coordinate
    x_coords = np.clip(x_coords, COORD_MIN, COORD_MAX).astype(int)
    y_coords = np.clip(y_coords, COORD_MIN, COORD_MAX).astype(int)

    # Pack instance as pandas dataframe
    nodes = np.arange(n_nodes) + 1

    return pd.DataFrame({
        "demands":  dict(zip(nodes, demands)),
        "earliest_arrivals": dict(zip(nodes, earliest_arrivals)),
        "x_coords": dict(zip(nodes, x_coords)),
        "y_coords": dict(zip(nodes, y_coords)),
    })
    

def solve_model(solver, model):
    """ Use a [solver] from Pyomo's solver factory to solve a Pyomo [model] and validate an optimal solution is found """
    start_time = perf_counter()
    result = solver.solve(model)
    solver_time = perf_counter() - start_time
    if result.solver.status != pyo.SolverStatus.ok or \
       result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError("Solver failed in %.2fs with status '%s' and termination condition '%s'" % 
                           (solver_time, result.solver.status, result.solver.termination_condition))
    print("Optimal solution found in %.2fs" % solver_time)
    
    
def plot_solution(coords, model):
    """ Plot the solution for an instance, provided a list of [coords] and a solved [model] """

    # Construct an adjacency matrix
    adjacency = {i: {j: int(model.arc[i, j]()) for j in model.nodes} for i in model.nodes}

    # Load matrix as networkx DiGraph
    G = nx.from_pandas_adjacency(pd.DataFrame(adjacency).T, create_using=nx.DiGraph)
    
    # Find the route of visited nodes between s and t
    route = nx.shortest_path(G, 1, model.n_nodes)

    # Create plot
    fig, ax = plt.subplots(1, 2, figsize=(24, 6))

    node_colors = ["green"] + ["blue"] * (coords.size - 2) + ["red"]

    # Plot instance and routes on a coordinate grid
    edge_labels = {(i, j): int(model.node_load[i]()) for i in model.nodes for j in model.nodes if model.arc[i, j]() == 1}

    nx.draw_networkx(G, ax=ax[0], pos=coords, nodelist=model.nodes, font_size=11, font_color="white", node_color=node_colors, alpha=0.75)
    nx.draw_networkx_edge_labels(G, ax=ax[0], pos=coords, edge_labels=edge_labels)

    # Plot the vehicle load during the route over time
    arrival_times = pd.Series([model.node_arrival[i]() for i in route], index=route)
    node_loads = pd.Series([model.node_load[i]() for i in route], index=route)

    ax[1].step(arrival_times, node_loads, where="post", lw=1, c="black")
    ax[1].scatter(arrival_times, node_loads, s=300, c=node_colors[:len(route)], alpha=0.75)
    for i in route:
        ax[1].annotate(i, (arrival_times[i], node_loads[i]), fontsize=11, color="white", ha="center", va="center")

    ax[1].set_xlabel("time (minutes)")
    ax[1].set_ylabel("load (x10^3 kg)")
    
    # Add solution scores as plot titles
    ax[0].set_title("Total travel time: %d" % model.total_travel_time())
    ax[1].set_title("Total fuel consumption: %d" % model.total_fuel_consumption())

    plt.show()