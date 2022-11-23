from itertools import product

import numpy as np
import pandas as pd


def sample_wind_vector(seed):
    """
    Samples a random wind vector.
    """
    rng = np.random.default_rng(seed)

    angle = np.pi * np.random.uniform(0, 0.5)
    wind_direction = np.array([np.cos(angle), np.sin(angle)])
    wind_speed = rng.integers(5, 25)

    return wind_direction * wind_speed


def compute_interferences(coords, wind_vector):
    """
    Computes the interference matrix.
    """
    size = len(coords)
    interferences = np.zeros((size, size))

    # For every two points that do not have the same coordinates, compute the interference
    for i, coord_i in enumerate(coords):
        for j, coord_j in enumerate(coords):
            if np.any(coord_i != coord_j):
                d_ij = coord_i - coord_j

                projection = np.dot(d_ij, wind_vector)
                strength = np.linalg.norm(projection) / np.linalg.norm(d_ij) ** 2

                if projection > 0:
                    interferences[i, j] = strength

    return interferences


def generate_instance(x, y, **kwargs):
    """
    Generates a random instance of grid size (x, y) with a fixed seed.
    """
    coords = np.array(list(product(range(x), range(y))))

    data = {
        "n_sites": x * y,
        "coords": coords,
        "production": 150,
        "min_distance": kwargs.get("min_distance", 2),
        "min_turbines": kwargs.get("min_turbines", 5),
        "max_turbines": 30,
    }

    return data


def read_elastic_net_data():
    """
    Returns a features matrix X and the target vector y.
    """
    wind_speed = pd.read_csv("data/elastic_net_wind_speed.csv").dropna()
    X = wind_speed[
        ["IND", "RAIN", "IND.1", "T.MAX", "IND.2", "T.MIN", "T.MIN.G"]
    ].values
    y = wind_speed["WIND"].values
    return X, y


def generate_wind_farm_data():
    """
    Computes all data needed for the wind farm part of the assignment.
    """
    grid_x_size = 10
    grid_y_size = 10
    n_locations = grid_x_size * grid_y_size
    instance = generate_instance(grid_x_size, grid_y_size)

    samples = 50
    wind_vectors = np.zeros((samples, 2))
    interference_matrices = np.zeros((samples, n_locations, n_locations))
    for i, seed in enumerate(range(samples)):
        wind_vectors[i] = sample_wind_vector(seed)
        interference_matrices[i] = compute_interferences(
            instance["coords"], wind_vectors[i]
        )

    wind_average = np.average(wind_vectors, axis=0)
    interference_average = compute_interferences(instance["coords"], wind_average)

    return instance, wind_average, interference_average, interference_matrices


def read_economic_dispatch_data():
    # Download the data
    nodes_df = pd.read_csv("data/economic_dispatch_nodes.csv", index_col=0)
    discrete_wind_df = pd.read_csv("data/economic_dispatch_wind_production.csv").T

    # Read data
    nodes = nodes_df.set_index("node_id").T.to_dict()
    discrete_wind = list(discrete_wind_df.to_dict().values())

    return nodes, discrete_wind
