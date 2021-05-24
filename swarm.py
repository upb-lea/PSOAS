"""Implementation of the Swarm class for the Particle Swarm Optimization."""

from operations import normal_distribution, uniform_distribution
from smt.sampling_methods import LHS
import numpy as np

from operations import random_hypersphere_draw, uniform_distribution


class Swarm():
    """
    TODO: Class docsting
    """

    def __init__(self, func, n_particles, dim, constr, options=None):
        """
        TODO: docstring
        """
        assert constr.shape == (dim, 2), f"Dimension of the particles ({dim}, 2) does not match the dimension of the constraints {constr.shape} !"

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
        sampling = LHS(xlimits=self.constr) # Set up latin hypercube sampling within given constraints
        self.position = sampling(self.n_particles)

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
        idx = np.argmin(self.pbest)
        gbest_position = self.pbest_position[idx, :]
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
        proj_lbest = self.position + c_2 * U_2 * (lbest_position - self.position) 

        center = (self.position + proj_pbest + proj_lbest) / 3

        r = np.linalg.norm(center - self.position, axis=1)

        offset = random_hypersphere_draw(r, self.dim)

        sample_points = center + offset
        
        omega = 1 / (2*np.log(2))
        self.velocity = omega * self.velocity + sample_points - self.position

    def compute_velocity(self):
        """
        TODO: docstring
        """
        if self.options['mode'] == 'SPSO2011':
            self._velocity_update_SPSO2011()
        else:
            raise NotImplementedError()
