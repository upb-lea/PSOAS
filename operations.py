import math

import numpy as np
from numpy.random import normal, random
import scipy

def uniform_distribution(num_particels, dimensions):
    return np.random.uniform(size=(num_particels, dimensions))

def normal_distribution(num_particels, dimensions):
    return np.random.normal(size=(num_particels, dimensions))

def random_hypersphere_draw(r, dim):
    """
    TODO: docstring
    """
    u = normal_distribution(r.shape[0], dim+2)
    u_norm = np.linalg.norm(u, axis=1)
    u /= u_norm[:,None]
    u *= r[:,None]
    return u[:,:dim]
