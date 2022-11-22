from itertools import product

import numpy as np


def sample_wind_vector(seed):
    """
    Samples a random wind vector.
    """
    rng = np.random.default_rng(seed)

    angle = np.pi * np.random.uniform(0, 0.5)
    wind_direction = np.cos(angle), np.sin(angle)
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


def generate_instance(x, y, seed, **kwargs):
    """
    Generates a random instance of grid size (x, y) with a fixed seed.
    """
    rng = np.random.default_rng(seed)

    coords = np.array(list(product(range(x), range(y))))
    interferences = compute_interferences(coords, sample_wind_vector(rng))

    data = {
        "n_sites": x * y,
        "coords": coords,
        "inteferences": interferences,
        "production": 150,
        "min_distance": kwargs.get("min_distance", 2),
        "min_turbines": kwargs.get("min_turbines", 5),
        "max_turbines": 30,
    }

    return data
