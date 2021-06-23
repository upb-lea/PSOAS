"""
Implementation of the optimizer class for the Particle Swarm Optimization. This class functions as the
optimizer and manager for the swarm, surrogates and databases.

Typical usage example:
    opt = Optimizer(func, n_particles, dimension, constraints)
    result = opt.optimize()
"""

import numpy as np
from numpy.core.fromnumeric import mean
import tableprint as tp

from psoas.swarm import Swarm
from psoas.surrogate import Surrogate


class Optimizer():
    """Optimizer class implementation.

    This class manages and updates the Swarm instance and any instances of surrogates and databases.
    It is designed to be used from the outside of the package to find the global optimum of a given 
    function. Furthermore it will hold functionality to evaluate the performance of the optimization 
    algorithm on benchmark-/testfunctions.
    """

    def __init__(self, func, n_particles, dim, constr, max_iter=100, options=None):
        """Creates and initializes an optimizer class instance.

        This function creates all class attributes which are necessary for an optimization process.
        It creates a Swarm instance which will be used in the optimization. Furthermore it creates
        some arrays which improve the computation time for the enforcing of the constraints.

        Args:
            func: The function whose global optimum is to be determined
            n_particles: The amount of particles which is used in the swarm
            dim: The dimension of the search-space
            constr: The constraints of the search-space with shape (dim, 2)
            max_iter: A integer value which determines the maximum amount of iterations in an 
                optimization call
            options: Options for the optimizer and swarm
        """
        # default options
        self.options = {'eps': 0.001,
                        'stalling_steps': 2,
                        'verbose': True,
                        'verbose_interval': 1,
                        'swarm_options': {'mode': 'SPSO2011', 
                                          'topology': 'global'}, 
                        'surrogate_options': {'surrogate_type': 'KRG',
                                              '3d_plot': False,
                                              'plotting_interval': 10}
                        }
        if options is not None:
            for key in options.keys():
                if key in ['swarm_options', 'surrogate_options']:
                    self.options[key].update(options[key])
                else:
                    self.options[key] = options[key]

        self.func = func
        self.max_iter = max_iter
        self.Swarm = Swarm(func, n_particles, dim, constr, self.options['swarm_options'])

        if 'surrogate_type' in options['surrogate_options'].keys():
            self.SurrogateModel = Surrogate(self.Swarm.position, self.Swarm.f_values, 
                                            self.options["surrogate_options"])

        self.constr_below = np.ones((n_particles, dim)) * constr[:, 0]
        self.constr_above = np.ones((n_particles, dim)) * constr[:, 1]
        self.velocity_reset = np.zeros((n_particles, dim))

    def update_swarm(self):
        """Updates the Swarm instance.

        The velocity update for the swarm is calculated here and the positions of all particles
        in the swarm are updated using this new velocity. The constraints are enforced by returning 
        any particle which left the valid search space, back into it. Lastly, the personal best 
        point for each particle is updated, if the function value at the new location is better than
        the previous personal best position.
        """
        self.Swarm.compute_velocity()
        self.enforce_constraints(check_position=False, check_velocity=True)

        self.Swarm.position = self.Swarm.position + self.Swarm.velocity

        self.enforce_constraints(check_position=True, check_velocity=False)

        # update pbest
        self.Swarm.f_values = self.Swarm.evaluate_function(self.Swarm.position)

        bool_decider = self.Swarm.pbest > self.Swarm.f_values
        self.Swarm.pbest[bool_decider] = self.Swarm.f_values[bool_decider]
        self.Swarm.pbest_position[bool_decider, :] = self.Swarm.position[bool_decider, :]

    def update_surrogate(self):
        """
        Docstring: TODO
        """
        self.SurrogateModel.update_data(self.Swarm.position, self.Swarm.f_values)
        self.SurrogateModel.fit_model()

    def optimize(self):
        """Main optimization routine.

        The swarm is updated until the maximum number of iteration is reached or the termination 
        condition is reached.

        Returns:
            A result dict, which holds the function value and position of the presumed global optimum,
            a list containing the function value history of the presumed global optimum per iteration,
            the amount of iterations used in the optimization process.
        """
        small_change_counter = 0

        results = {"gbest_list":[], "iter": None}
        for i in range(self.max_iter):
            prior_pbest = self.Swarm.pbest.copy()

            self.update_swarm()

            if hasattr(self, 'SurrogateModel') and self.options['surrogate_options']['3d_plot'] and i%10 == 0:
                self.update_surrogate()
                self.SurrogateModel.plotter_2d()

            gbest, gbest_position = self.Swarm.compute_gbest()
            results['gbest_list'].append(gbest)

            mean_squared_change = np.linalg.norm(prior_pbest - self.Swarm.pbest)
            if mean_squared_change < self.options['eps']:
                small_change_counter += 1
            else:
                small_change_counter = 0
            
            if self.options['verbose']:
                self.print_iteration_information(i, gbest)

            if small_change_counter >= self.options['stalling_steps']:
                results['iter'] = i+1
                break

        if self.options['verbose']:
            print(tp.bottom(len(self.headers), width=20))
            print('\n')

        results['x_opt'] = gbest_position
        results['func_opt'] = gbest
        if results['iter'] == None:
            results['iter'] = self.max_iter
        return results

    def enforce_constraints(self, check_position, check_velocity):
        """Enforces the constraints of the valid search space.

        Any particle which left the valid search space is moved back into it. Furthermore
        the velocity which brought the particle out of the valid search space is put to 
        zero.
        """
        if check_position:
            bool_below = self.Swarm.position < self.Swarm.constr[:, 0]
            bool_above = self.Swarm.position > self.Swarm.constr[:, 1]

            self.Swarm.position[bool_below] = self.constr_below[bool_below]
            self.Swarm.position[bool_above] = self.constr_above[bool_above]
            self.Swarm.velocity[bool_below] = self.velocity_reset[bool_below]
            self.Swarm.velocity[bool_above] = self.velocity_reset[bool_above]

        if check_velocity:
            bool_below = self.Swarm.velocity < self.Swarm.constr[:, 0]
            bool_above = self.Swarm.velocity > self.Swarm.constr[:, 1]

            self.Swarm.velocity[bool_below] = self.constr_below[bool_below]
            self.Swarm.velocity[bool_above] = self.constr_above[bool_above]

    def print_iteration_information(self, idx, gbest):
        if idx == 0:
            print('\n', 'Options:')
            print(self.options, '\n')

            self.headers = ['idx', 'gbest', 'mean_pbest', 'var_pbest']
            print(tp.header(self.headers, width=20))

        elif idx % self.options['verbose_interval'] == 0:
            mean_pbest = np.mean(self.Swarm.pbest)
            var_pbest = np.var(self.Swarm.pbest)

            data = [idx, gbest, mean_pbest, var_pbest]
            assert len(data) == len(self.headers)

            print(tp.row(data, width=20))
