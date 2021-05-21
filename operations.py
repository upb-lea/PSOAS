import math

import numpy as np
from numpy.random import normal, random
import scipy

def uniform_distribution(num_particels, dimensions):
    return np.random.uniform(0, 1, (num_particels, dimensions))

def normal_distribution(num_particels, dimensions):
    return np.random.normal(0, 1, (num_particels, dimensions))
