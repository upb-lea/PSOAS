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

def calc_max_iter(max_f_eval, n_particles):
    max_iter = np.floor((max_f_eval - n_particles)/n_particles)
    return max_iter.astype('int')

class counting_function():
    def __init__(self, function):
        self.eval_count = 0
        self.function = function

    def __call__(self, x):
        self.eval_count += 1
        return self.function(x)


class counting_function_cec2013_single(counting_function):
    def __call__(self, x):
        length = x.shape[0]
        self.eval_count += length

        res = np.zeros(length)
        for idx in range(length):
            res[idx] = self.function(x[idx, :])
        return res


class DataBuffer:
    
    def __init__(self, dim, n_particles, n_slots=4):

        self.position_buffer = np.zeros((n_slots * n_particles, dim), dtype=np.float32)
        self.f_val_buffer = np.zeros((n_slots * n_particles, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = n_slots
        self.n_particles = n_particles
    
    def store(self, positions, f_vals):

        start = self.ptr * self.n_particles
        end = (self.ptr+1) * self.n_particles

        self.position_buffer[start:end] = positions.copy()
        self.f_val_buffer[start:end] = f_vals.copy()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def fetch(self):

        positions = self.position_buffer[:self.size*self.n_particles]
        f_vals = self.f_val_buffer[:self.size*self.n_particles]
        return positions, f_vals
