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


        for idx in range(self.Swarm.n_particles):
            if self.Swarm.pbest[idx] <= func_eval[idx]:
                continue
            if self.Swarm.pbest[idx] > func_eval[idx]:
                #print('Change')
                self.Swarm.pbest[idx] = func_eval[idx]
                self.Swarm.pbest_position[idx, :] = self.Swarm.position[idx, :]
                #print(self.pbest_position[idx, :])

    def optimize(self):
        """
        TODO: docstring
        """
        for i in range(self.max_iter):
            self.update_swarm()