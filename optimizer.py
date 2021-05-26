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

    def enforce_constraints(self):
        positions = self.Swarm.position
        constr = self.Swarm.constr
        velocity = self.Swarm.velocity


        bool_below = positions < self.Swarm.constr[:, 0]
        bool_above = positions > self.Swarm.constr[:, 1]

        for dim in range(self.Swarm.dim):
            positions[bool_below[:, dim], dim] = self.Swarm.constr[dim, 0]
            positions[bool_above[:, dim], dim] = self.Swarm.constr[dim, 1]

            velocity[bool_below[:, dim], dim] = 0
            velocity[bool_above[:, dim], dim] = 0
