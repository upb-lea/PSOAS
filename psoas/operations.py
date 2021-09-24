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


class TimeDataBuffer:
    
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


class ValueDataBuffer:
    
    def __init__(self, dim, buffer_size):

        self.position_buffer = np.zeros((buffer_size, dim), dtype=np.float32)
        self.f_val_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = buffer_size
    
    def store(self, positions, f_vals):
        length = positions.shape[0]        

        for i in range(length):
            
            # check if the position is already stored in the buffer
            already_in_buffer = (positions[i] == self.position_buffer).all(axis=-1).any()
            if already_in_buffer:
                continue
            
            # if the buffer is not full yet, the new point is added 
            # regardless of its value
            if self.size < self.max_size:
                self.position_buffer[self.ptr] = positions[i]
                self.f_val_buffer[self.ptr] = f_vals[i]
                self.ptr += 1
                self.size += 1
            
            # if the buffer is full, check if the new point is better
            # than the current worst point 
            else:
                idx_max = np.argmax(self.f_val_buffer)
                f_val_max = self.f_val_buffer[idx_max]
                if f_vals[i] < f_val_max:
                    self.position_buffer[idx_max] = positions[i]
                    self.f_val_buffer[idx_max] = f_vals[i]
    
    def fetch(self):
        positions = self.position_buffer[:self.size]
        f_vals = self.f_val_buffer[:self.size]
        return positions, f_vals