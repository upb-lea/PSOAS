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
            self.options = {"mode": 'SPSO2011', "topology":'global'}
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
        self.pbest_position = self.position
        self.pbest = self.func(self.position)

    def _use_topology():
        """
        TODO: docstring
        """
        raise NotImplementedError()

    def compute_gbest(self):
        """Returns the global optimum found by any of the particles."""
        idx = np.argmax(self.pbest)
        gbest_position = self.pbest_position[idx]
        gbest = self.pbest[idx]
        return gbest, gbest_position

    def compute_lbest(self):
        """
        Returns the local optimum for each particle depending on the topology
        specified in the options.
        """
        if self.options['topology'] == 'global':
            gbest, gbest_position = self.compute_gbest()
            ones = np.ones(self.n_particles)
            return gbest * ones, ones[:, None] @ gbest_position[None, :]  
        else:
            raise NotImplementedError()
    
    def _velocity_update_SPSO2011(self):
        """
        TODO: docstring, SPSO2011(ZambranoBigiarini2013)
        """
        lbest = self.compute_lbest()
        

    def compute_velocity(self):
        """
        TODO: docstring
        """
        if options['mode'] == 'SPSO2011':
            _velocity_update_SPSO2011()
        else:
            raise NotImplementedError()

    def update_swarm():
        """
        TODO: docstring
        """
        raise NotImplementedError()
