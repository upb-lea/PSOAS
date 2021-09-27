"""Collection of basic utillity functions."""

import numpy as np


def random_hypersphere_draw(r, dim):
    """Uniform sampling from dim-dimensional hyperspheres.

    The goal is to sample a dim-dimensional array for each particle. One sample is
    drawn uniformly from a hypersphere where the radius corresponds to one element of r.
    The exact explanation can be found in the paper Völker2017
    (doi: 10.13140/RG.2.2.15829.01767/1).

    Args:
        r: Array of shape (n_particles,) containing the different radii
        dim: The dimension of the search-space

    Returns:
        u: Array of shape (n_particles, dim) uniformly drawn samples from a
            dim-dimensional hypersphere
    """
    u = np.random.normal(size=(r.shape[0], dim+2))
    u_norm = np.linalg.norm(u, axis=1)
    u /= u_norm[:,None]
    u *= r[:,None]
    return u[:,:dim]


def calc_max_iter(max_f_eval, n_particles):
    """Calculate the maximum interations.

    For some applications, a fixed bugdet of functions evaluations must
    be considered. This function calculates the maximum possible iterations
    given the budget and the number of particles.
    
    Args:
        max_f_eval: Maximum number of function evaluations
        n_particles: The number of particles used in the swarm

    Returns:
        max_iter: Maximum number of interations
    """
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
    """Implementation of the time data buffer.

    This class implements a ringbuffer that works according to the first in, first out
    (FIFO) principle. The positions and associated function values are stored for each
    iteration, where the number of iterations can be set.

    Attributes:
        position_buffer: Array of shape ((n_slots * n_particles), dim) containing the
            positions of the last n_slots-iterations
        f_val_buffer: Array of shape ((n_slots * n_particles, 1) containing the function
            values of the last n_slots-iterations

    """
    def __init__(self, dim, n_particles, n_slots=4):
        """Creates and initializes a time data buffer class instance.
        
        Args:
            dim: The dimension of the search-space
            n_particles: The number of particles used in the swarm
            n_slots: The number of slots used for storage
        """

        self.position_buffer = np.zeros((n_slots * n_particles, dim), dtype=np.float32)
        self.f_val_buffer = np.zeros((n_slots * n_particles, 1), dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._max_size = n_slots
        self._n_particles = n_particles
    
    def store(self, positions, f_vals):
        """Store the positions and corresponding function values.

        Args:
            positions: The positions of the particles in dim-dimensional space with
            shape (n_particles, dim)
            f_vals: The function values of the particles with shape (n_particles, 1)
        """
        start = self._ptr * self._n_particles
        end = (self._ptr+1) * self._n_particles

        self.position_buffer[start:end] = positions.copy()
        self.f_val_buffer[start:end] = f_vals.copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def fetch(self):
        """Fetch the positions and corresponding function values.

        Returns:
            postions: Array of shape ((n_slots * n_particles), dim) containing the
                positions of the buffer
            f_vals: Array of shape ((n_slots * n_particles), 1) containing the
                function values of the buffer

        """
        positions = self.position_buffer[:self._size*self._n_particles]
        f_vals = self.f_val_buffer[:self._size*self._n_particles]
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
