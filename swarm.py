"""
author: Hendrik Vater (hvater@mail.upb.de)

Implementation of the Swarm class for the Particle Swarm Optimization.
"""

import numpy as np

class Swarm():
    """
    TODO: Class docsting
    """

    def __init__(self, func, n_particles, dim, constr, options=None):
        """
        TODO: docstring
        """
        assert constr.shape == (2, dim)

        if options is None:
            self.options = {"mode": 'SPSO2011'}
        else:
            self.options = options

        self.func = func
        self.n_particles = n_particles
        self.dim = dim
        self.constr = constr
        self._calculate_initial_values()

    def _calculate_initial_values(self):
        """
        TODO: docstring, Hypercube-sampling
        """
        self.position = np.random.uniform(0, 1, (self.n_particles, self.dim))
        self.velocity = np.random.uniform(0, 1, (self.n_particles, self.dim))
        self.pbest = self.func(self.position)

    def _use_topology():
        """
        TODO: docstring
        """
        raise NotImplementedError()

    def compute_gbest(self):
        """Returns the global optimum found by any of the particles."""
        return np.max(self.pbest)

    def compute_lbest(self):
        """
        Returns the local optimum for each particle depending on the topology
        specified in the options.
        """
        raise NotImplementedError()
    
    def compute_velocity(self):
        """
        TODO: docstring, SPSO2011(ZambranoBigiarini2013)
        """
        if options['mode'] == 'SPSO2011':
            raise NotImplementedError()
        else:
            raise NotImplementedError()