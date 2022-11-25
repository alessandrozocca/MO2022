from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sample_wind_vector(seed):
    """
    Samples a random wind vector.
    """
    rng = np.random.default_rng(seed)

    target_angle = 0.4
    width = 0.65
    angle = np.pi * rng.uniform(target_angle - width, target_angle + width)
    wind_direction = np.array([np.cos(angle), np.sin(angle)])
    wind_speed = 33 * rng.uniform() * max(np.cos(angle - target_angle), 0)

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

                projection = 20 * np.dot(d_ij, wind_vector)
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
        "min_distance": kwargs.get("min_distance", 1.9),
        "min_turbines": kwargs.get("min_turbines", 3),
        "max_turbines": 6,
    }

    return data


def read_elastic_net_data():
    """
    Returns a features matrix X and the target vector y.
    """
    wind_speed = pd.read_csv(
        "https://gist.githubusercontent.com/leonlan/dc606eee560edde18fd47339b7ad2954/raw/5ef38f264134ddd1be0331202616c78dd75be624/wind_speed.csv"
    ).dropna()
    X = wind_speed[
        ["IND", "RAIN", "IND.1", "T.MAX", "IND.2", "T.MIN", "T.MIN.G"]
    ].values
    y = wind_speed["WIND"].values
    return X, y


def generate_wind_farm_data(n_samples=45, x=6, y=6, seed=0):
    """
    Computes all data needed for the wind farm part of the assignment.
    """
    grid_x = x
    grid_y = y

    n_locations = grid_x * grid_y
    instance = generate_instance(grid_x, grid_y)

    samples = n_samples
    wind_vectors = np.zeros((samples, 2))
    interference_matrices = np.zeros((samples, n_locations, n_locations))

    for idx in range(samples):
        wind_vectors[idx] = sample_wind_vector(seed * n_samples + idx)
        interference_matrices[idx] = compute_interferences(
            instance["coords"], wind_vectors[idx]
        )

    return instance, interference_matrices


def read_economic_dispatch_data():
    nodes_df = pd.read_csv(
        "https://gist.githubusercontent.com/leonlan/8145e4477dabe97705c60aa4d55363f5/raw/6ab2d382a0634125aa25f469faa1d7a03afb8596/nodes.csv",
        index_col=0,
    )[["node_id", "d", "p_min", "p_max", "c_var"]]

    wind_production_samples_df = pd.read_csv(
        "https://gist.githubusercontent.com/leonlan/8145e4477dabe97705c60aa4d55363f5/raw/7816951386b4cdd2b624b0c4a34a6c8b66bc1dc8/discrete_wind.csv"
    ).T

    # Read data
    nodes = nodes_df.set_index("node_id").T.to_dict()
    wind_production_samples = list(wind_production_samples_df.to_dict().values())
    wind_production_samples = [sum(d.values()) for d in wind_production_samples]

    return nodes, wind_production_samples


def plot_wind_farm(instance, solution, **kwargs):
    coords = instance["coords"]
    min_distance = instance["min_distance"]

    _, ax = plt.subplots(figsize=[10, 10])

    ax.scatter(coords[:, 0], coords[:, 1], s=100, label="site")

    for idx in range(len(coords)):
        margin = 0.05
        ax.annotate(idx, (coords[idx, 0] + margin, coords[idx, 1] + margin))

    turbine_coords = coords[solution]
    ax.scatter(*turbine_coords.T, s=100, label="turbine")

    for (x, y) in turbine_coords:
        cir = plt.Circle((x, y), min_distance / 2, color="r", fill=False)
        ax.add_patch(cir)

    ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
    ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)

    ax.set_title(f"Wind farm layout\nSolution {solution}")
    ax.legend()
