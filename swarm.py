"""
author: Hendrik Vater (hvater@mail.upb.de)

Implementation of the Swarm class for the Particle Swarm Optimization.
"""

from operations import normal_distribution, uniform_distribution
from smt.sampling_methods import LHS
from pyDOE2 import lhs
import numpy as np

class Swarm():
    """
    TODO: Class docsting
    """

    def __init__(self, func, n_particles, dim, constr=None, options=None):
        """
        TODO: docstring
        """
        # assert constr.shape == (2, dim)

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
        print('This is self.constr ->', self.constr)
        if self.constr != None:
            sampling = LHS(xlimits=self.constr)
            self.position = sampling(self.n_particles)
        else:
            self.position = lhs(self.dim, self.n_particles)

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
        lbest, lbest_position = self.compute_lbest()
        # gbest, gbest_position = self.compute_gbest()
        
        c_1, c_2 = np.ones(2) * 0.5 + np.log(2)
       
        U_1 = uniform_distribution(self.n_particles, self.dim)
        U_2 = uniform_distribution(self.n_particles, self.dim)

        proj_pbest = self.position + c_1 * U_1 * (self.pbest_position - self.position)
        proj_lbest = self.position + c_2 * U_2 * (self.lbest_position - self.position) 

        G = (self.position + proj_pbest + proj_lbest) / 3

        radius = np.linalg.norm(G - self.position, axis=1)

        u = normal_distribution(self.n_particles, self.dim+2)

        weighted_draw = radius[:, None]*u # Need to evaluated, not sure if this work right.

        norm = np.sqrt(np.sum(weighted_draw**2))

        weighted_draw = weighted_draw/norm

        rand_sampling = weighted_draw[:,:self.dim]


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
