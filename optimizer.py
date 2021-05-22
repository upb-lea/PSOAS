"""
Implementation of the optimizer class for the Particle Swarm Optimization. This class functions as the
optimizer and manager for the swarm, surrogates and databases.
"""

import numpy as np

from swarm import Swarm


class Optimizer():
    """
    TODO: Class docsting
    """

    def __init__(self, func, n_particles, dim, constr, max_iter=100, options=None):
        """
        TODO: docstring
        """
        self.func = func
        self.max_iter = max_iter
        self.Swarm = Swarm(func, n_particles, dim, constr)

    def __call__(self):
        """
        TODO: docstring
        """
        raise NotImplementedError()

    def update_swarm(self):
        """
        TODO: docstring
        """
        self.Swarm.compute_velocity()
        self.Swarm.position = self.Swarm.position + self.Swarm.velocity

        # update pbest
        func_eval = self.Swarm.func(self.Swarm.position)

        bool_decider = self.Swarm.pbest > func_eval
        self.Swarm.pbest[bool_decider] = func_eval[bool_decider]
        self.Swarm.pbest_position[bool_decider, :] = self.Swarm.position[bool_decider, :]

    def optimize(self):
        """
        TODO: docstring
        """
        gbest_list = []
        for i in range(self.max_iter):
            self.update_swarm()
            gbest, gbest_position = self.Swarm.compute_gbest()
            gbest_list.append(gbest)
        return gbest_list
