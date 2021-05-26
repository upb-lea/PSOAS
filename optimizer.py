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

        self.constr_below = np.ones((n_particles, dim)) * constr[:, 0]
        self.constr_above = np.ones((n_particles, dim)) * constr[:, 1]
        self.velocity_reset = np.zeros((n_particles, dim))

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

        self.enforce_constraints()

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

    def enforce_constraints(self):
        """
        TODO: docstring
        """
        bool_below = self.Swarm.position < self.Swarm.constr[:, 0]
        bool_above = self.Swarm.position > self.Swarm.constr[:, 1]

        self.Swarm.position[bool_below] = self.constr_below[bool_below]
        self.Swarm.position[bool_above] = self.constr_above[bool_above]
        self.Swarm.velocity[bool_below] = self.velocity_reset[bool_below]
        self.Swarm.velocity[bool_above] = self.velocity_reset[bool_above]
