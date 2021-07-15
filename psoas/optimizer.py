"""
Implementation of the optimizer class for the Particle Swarm Optimization. This class functions as the
optimizer and manager for the swarm, surrogates and databases.

Typical usage example:
    opt = Optimizer(func, n_particles, dimension, constraints)
    result = opt.optimize()
"""
import numpy as np
import matplotlib.pyplot as plt
import pprint
import tableprint as tp
from tqdm import tqdm

from psoas.operations import normal_distribution, counting_function, counting_function_cec2013_single
from psoas.swarm import Swarm
from psoas.surrogate import Surrogate


class Optimizer():
    """Optimizer class implementation.

    This class manages and updates the Swarm instance and any instances of surrogates and databases.
    It is designed to be used from the outside of the package to find the global optimum of a given 
    function. Furthermore it will hold functionality to evaluate the performance of the optimization 
    algorithm on benchmark-/testfunctions.
    """

    def __init__(self, func, n_particles, dim, constr, max_iter, **kwargs):
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
        self.options = {'eps_abs': 0.001,
                        'eps_rel': 0.001,
                        'stalling_steps': 10,
                        'verbose': False,
                        'verbose_interval': 50,
                        'do_plots': False,
                        'swarm_options': {'mode': 'SPSO2011', 
                                          'topology': 'global',
                                          '3d_plot': False,
                                          'create_gif': False}, 
                        'surrogate_options': {'surrogate_type': 'GP',
                                              'use_surrogate': True,
                                              '3d_plot': False,
                                              'interval': 10,
                                              'prediction_mode': 'standard'}
                        }

        for key, value in kwargs.items():
            if type(value) is dict:
                for inner_key, inner_value in value.items():
                    if inner_key not in self.options[key]:
                        raise NameError(f'The key "{inner_key}" does not exist in the dict.')
                    
                    else:
                        self.options[key][inner_key] = inner_value
                    
            elif key in self.options:
                self.options[key] = value
            
            else:
                raise NameError(f'The key "{key}" does not exist in the dict.')

        self.func = counting_function_cec2013_single(func)
        self.dim = dim
        self.max_iter = max_iter
        self.Swarm = Swarm(self.func, n_particles, dim, constr, self.options['swarm_options'])

        if 'surrogate_type' in self.options['surrogate_options'].keys():
            self.SurrogateModel = Surrogate(self.Swarm.position, self.Swarm.f_values, 
                                            self.options["surrogate_options"])

        self.Swarm.no_change_in_gbest = False
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
        TODO: docstring
        """
        self.SurrogateModel.update_data(self.Swarm.position, self.Swarm.f_values)
        self.SurrogateModel.fit_model()

    def use_surrogate_prediction(self):
        """
        TODO: docstring
        """
        if self.options['surrogate_options']['prediction_mode'] == 'standard':
            prediction = self.SurrogateModel.get_prediction_point(self.Swarm.constr)
            prediction_point = prediction[0][0]
            f_val_at_pred = self.func(prediction_point[None,:])

            self.SurrogateModel.update_data(prediction_point[None,:], f_val_at_pred)

            idx = np.argmax(self.Swarm.pbest)

            self.Swarm.position[idx] = prediction_point
            self.Swarm.f_values[idx] = f_val_at_pred
            self.Swarm.velocity[idx] = normal_distribution(1, self.dim)

            if f_val_at_pred < self.Swarm.pbest[idx]:
                self.Swarm.pbest[idx] = f_val_at_pred
                self.Swarm.pbest_position[idx] = prediction_point


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

        results = {"iter": None}

        if self.options['do_plots']:
            results["gbest_list"] = []
            results["mean_pbest_list"] = []
            results["var_pbest_list"] = []
            results["n_fun_eval_list"] = []

        for i in range(self.max_iter):
            prior_pbest = self.Swarm.pbest.copy()
            prior_gbest, _  = self.Swarm.compute_gbest()

            self.update_swarm()

<<<<<<< HEAD
            self.Swarm.plotter()

            if (hasattr(self, 'SurrogateModel') and self.options['surrogate_options']['use_surrogate']
=======
            if (self.options['surrogate_options']['use_surrogate']
>>>>>>> c79734ac4bb48b5b8df6fe8f693e0a52772c314e
                and i % self.options['surrogate_options']['interval'] == 0):

                assert hasattr(self, 'SurrogateModel')
                self.update_surrogate()

                self.use_surrogate_prediction()

                if self.options['surrogate_options']['3d_plot']:
                    self.SurrogateModel.plotter_3d()

            gbest, gbest_position = self.Swarm.compute_gbest()
            self.Swarm.no_change_in_gbest = (prior_gbest - gbest == 0)

            mean_squared_change = np.linalg.norm(prior_pbest - self.Swarm.pbest)
            norm_mean_squared_change = np.linalg.norm((prior_pbest - self.Swarm.pbest) / prior_pbest)
            if mean_squared_change < self.options['eps_abs']:
                small_change_counter += 1
            elif norm_mean_squared_change < self.options['eps_rel']:
                small_change_counter += 1
            else:
                small_change_counter = 0
            
            if self.options['do_plots']:
                results['gbest_list'].append(gbest)
                results['mean_pbest_list'].append(np.mean(self.Swarm.pbest))
                results['var_pbest_list'].append(np.var(self.Swarm.pbest))
                results["n_fun_eval_list"].append(self.func.eval_count)

            if self.options['verbose']:
                self.print_iteration_information(i, gbest)

            if small_change_counter >= self.options['stalling_steps']:
                results['iter'] = i+1
                results['term_flag'] = 2
                break

        if results['iter'] == None:
            results['iter'] = self.max_iter
            results['term_flag'] = 1

        if self.options['verbose']:
            print(tp.bottom(len(self.headers), width=20))
            print('\n')

        if self.options['do_plots']:
            self.plot_results(results)

        if self.options['swarm_options']['create_gif']:
            self.Swarm.create_gif()
        
        results['mean_pbest'] = np.mean(self.Swarm.pbest)
        results['var_pbest'] = np.var(self.Swarm.pbest)
        results['x_opt'] = gbest_position
        results['func_opt'] = gbest
        results['n_fun_evals'] = self.func.eval_count

        return results


    def stalling(self):
        raise NotImplementedError


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
            pprint.pprint(self.options)
            print()

            self.headers = ['idx', 'gbest', 'mean_pbest', 'var_pbest']
            print(tp.header(self.headers, width=20))

        elif idx % self.options['verbose_interval'] == 0:
            mean_pbest = np.mean(self.Swarm.pbest)
            var_pbest = np.var(self.Swarm.pbest)

            data = [idx, gbest, mean_pbest, var_pbest]
            assert len(data) == len(self.headers)

            print(tp.row(data, width=20))

    def plot_results(self, results):
        gbest = np.array(results['gbest_list'])
        mean_pbest = np.array(results['mean_pbest_list'])
        var_pbest = np.array(results['var_pbest_list'])
        x = np.array(results['n_fun_eval_list'])

        fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(12,12))
        axs[0].plot(x, gbest, color='orange')
        axs[0].set_ylabel('gbest fval')
        axs[1].plot(x, mean_pbest, color='tab:blue')
        axs[1].fill_between(x, mean_pbest - np.sqrt(var_pbest), mean_pbest + np.sqrt(var_pbest),
                            color='tab:blue', alpha=0.2)
        

        axs[1].set_ylabel('mean +- std pbest fval')
        axs[1].set_xlabel('function evaluations')

        fig.show()