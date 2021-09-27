"""Implementation of the optimizer class for the Particle Swarm Optimization. This class functions as the
optimizer and manager for the swarm and surrogate.

Typical usage example:
    opt = Optimizer(func, n_particles, dimension, constraints, max_iter)
    result = opt.optimize()
"""
import numpy as np
import matplotlib.pyplot as plt
import pprint
from tqdm import tqdm
import tableprint as tp

from psoas.utils import counting_function, counting_function_cec2013_single
from psoas.swarm import Swarm
from psoas.surrogate import Surrogate


class Optimizer():
    """Optimizer class implementation.

    This class manages and updates the swarm and surrogate instances. It is designed to be used from 
    the outside of the package to find the global optimum of a given function. It holds some functionalities
    to monitor an optimization during its runtime with a table that is iteratively updated and to illustrate
    the optimization after it is finished using convergence plots for the global best and the mean and
    standard deviation of the personal bests of the swarm.

    Attributes:
        options: A dict containing all options which can be set for the optimization
        func: The function to be optimized
        n_particles: The number of particles in the swarm
        dim: The dimension of the search-space
        max_iter: A integer value which determines the maximum amount of iterations in an 
                optimization call
        max_func_evals: The maximum amount of function evaluations in an optimization call
        Swarm: A swarm instance which is used for the PSO, refer to the documentation of this
            class for further information
        SurrogateModel: a surrogate instance which is used in the PSO, refer to the 
            documentation of this class for further information (Only if the use_surrogate
            option is set to be True) 
        current_prediction: The last point that was proposed by the surrogate (Only 
            necessary for surrogate prediction modes center_of_gravity and 
            shifting_center)
        worst_idx: The particle index at which a proposed point is stored (Only for
            standard surrogate prediction mode)
        worst_indices: The particle indices at which proposed points are stored
            (Only for standard_m surrogate prediction mode)
        other_indices: The particle indices that complement worst_indices (Only for
            standard_m surrogate prediction mode)
    """

    def __init__(self, func, n_particles, dim, constr, max_iter, max_func_evals=None, **kwargs):
        """Creates and initializes an optimizer class instance.

        This function creates all class attributes which are necessary for an optimization process.
        It creates a Swarm instance which will be used in the optimization.

        Args:
            func: The function to be optimized
            n_particles: The number of particles in the swarm
            dim: The dimension of the search-space
            constr: The constraints of the search-space with shape (dim, 2)
            max_iter: A integer value which determines the maximum amount of iterations in an 
                optimization call
            max_func_evals: The maximum amount of function evaluations in an optimization call
            **kwargs: The remaining keywords constitute the options which differ from the 
                default values, alternatively one can insert and options dict with **options
                (This options dict does not need to be complete).
        """
        self.options = self._fetch_default_options()
        self._update_options(kwargs)
        
        self.func = counting_function_cec2013_single(func)
        self.n_particles = n_particles
        self.dim = dim
        self.max_iter = max_iter
        self.max_func_evals = max_func_evals

        self._options_checker()

        self.Swarm = Swarm(self.func, n_particles, dim, constr, self.options['swarm_options'], 
                           self.options['surrogate_options'])

        if self.options['surrogate_options']['use_surrogate']:
            self.SurrogateModel = Surrogate(self.Swarm.positions, self.Swarm.f_values, 
                                            self.options["surrogate_options"])

        self.Swarm.no_change_in_gbest = False

    def _fetch_default_options(self):
        """Returns a dict containing the default options."""

        default_options = {'eps_abs': 0.0,
                           'eps_rel': 0.0,
                           'stalling_steps': 10,
                           'verbose': False,
                           'verbose_interval': 1,
                           'do_plots': False,
                           'swarm_options': {'mode': 'SPSO2011', 
                                             'topology': 'global',
                                             'contour_plot': False,
                                             'create_gif': False},
                           'surrogate_options': {'surrogate_type': 'GP',
                                                 'use_surrogate': True,
                                                 'use_buffer': True,
                                                 'buffer_type': 'time',
                                                 'n_slots': 4,
                                                 '3d_plot': False,
                                                 'interval': 1,
                                                 'm': 5,
                                                 'prediction_mode': 'standard',
                                                 'prioritization': 0.2}
                           }
        return default_options

    def _update_options(self, kwargs):
        """Updates the options dict with the input parameters."""
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

    def _options_checker(self):
        if self.options['surrogate_options']['use_surrogate']:
            if self.options['surrogate_options']['prediction_mode'] == 'standard_m':
                assert self.options['surrogate_options']['m'] <= self.n_particles, 'm must be less than or equal to the number of particles.'

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

            if (self.options['surrogate_options']['use_surrogate']
                and i % self.options['surrogate_options']['interval'] == 0
               ):

                assert hasattr(self, 'SurrogateModel')
                if i > 0:
                    self.SurrogateModel.update_surrogate(self.Swarm.positions, self.Swarm.f_values)

                if self.options['surrogate_options']['3d_plot']:
                    self.SurrogateModel.plotter_3d()

                self.use_surrogate_prediction()

            self.update_swarm()

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

            if self.options['swarm_options']['contour_plot']:
                self.Swarm.swarm_plotter.plot(self.Swarm.positions, self.Swarm.velocities, 
                                              self.Swarm.swarm_options['create_gif'])

            if self.options['verbose']:
                self.print_iteration_information(i, gbest)

            if small_change_counter >= self.options['stalling_steps']:
                results['iter'] = i+1
                results['term_flag'] = 2
                break

            if self.check_max_func_evals():
                results['iter'] = i+1
                results['term_flag'] = 0
                break

        if results['iter'] == None:
            results['iter'] = self.max_iter
            results['term_flag'] = 1

        if self.options['verbose']:
            print(tp.bottom(len(self._headers), width=20))
            print('\n')

        if self.options['do_plots']:
            self.plot_results(results)

        if self.options['swarm_options']['contour_plot'] and self.options['swarm_options']['create_gif']:
            self.Swarm.swarm_plotter.create_gif()
        
        results['mean_pbest'] = np.mean(self.Swarm.pbest)
        results['var_pbest'] = np.var(self.Swarm.pbest)
        results['x_opt'] = gbest_position
        results['func_opt'] = gbest
        results['n_fun_evals'] = self.func.eval_count

        return results
    
    def update_swarm(self):
        """Updates the Swarm instance.

        The velocity update for the swarm is calculated here and the positions of all particles
        in the swarm are updated using this new velocity. The constraints are enforced by returning 
        any particle which left the valid search space, back into it. Lastly, the personal best 
        point for each particle is updated, if the function value at the new location is better than
        the previous personal best position.
        """
        if not self.options['surrogate_options']['use_surrogate']:
            self.Swarm.update()
        else:
            if (self.options['surrogate_options']['prediction_mode'] == 'center_of_gravity' or 
                self.options['surrogate_options']['prediction_mode'] == 'shifting_center'
            ):
                self.Swarm.update(self.current_prediction)
            elif self.options['surrogate_options']['prediction_mode'] == 'standard':
                self.Swarm.update(worst_idx=self.worst_idx)
            elif self.options['surrogate_options']['prediction_mode'] == 'standard_m':
                self.Swarm.update(worst_indices=self.worst_indices, other_indices=self.other_indices)

    def enforce_constraints(self, check_position, check_velocity):
        """Wraps the enforce_constraints method from the swarm."""
        self.Swarm.enforce_constraints(check_position, check_velocity)

    def use_surrogate_prediction(self):
        """
        This function handles the different prediction methods.
        """
        if self.options['surrogate_options']['prediction_mode'] == 'standard':
            pos_prediction, f_val_prediction = self.SurrogateModel.get_prediction_point(self.Swarm.constr)
            self.worst_idx = np.argmax(self.Swarm.pbest)
            self.Swarm.positions[self.worst_idx] = pos_prediction

        if self.options['surrogate_options']['prediction_mode'] == 'standard_m':
            self.worst_indices, self.other_indices = self.SurrogateModel.use_standard_m_prediction(self.Swarm)

        elif (self.options['surrogate_options']['prediction_mode'] == 'center_of_gravity' or 
              self.options['surrogate_options']['prediction_mode'] == 'shifting_center'
             ):
            pos_prediction, f_val_prediction = self.SurrogateModel.get_prediction_point(self.Swarm.constr)
            self.current_prediction = pos_prediction

    def check_max_func_evals(self):
        """Returns true if the optimization should be terminated due to too many function evaluations.
        
        This is always done such that the amount of function evaluations is lower than the given maximum.
        But it is not ensured that the amount of function evaluations is fully exhausted. If the next 
        iteration would lead to too many function evaluations, the iteration would not be started and
        the optimization terminates.
        """
        if self.max_func_evals is None:
            return False
  
        func_evals_iter = self.Swarm.n_particles

        if self.func.eval_count + func_evals_iter > self.max_func_evals:
            return True
        else:
            return False

    def print_iteration_information(self, idx, gbest):
        if idx == 0:
            print('\n', 'Options:')
            pprint.pprint(self.options)
            print()

            self._headers = ['idx', 'gbest', 'mean_pbest', 'var_pbest']
            print(tp.header(self._headers, width=20))

        if idx % self.options['verbose_interval'] == 0:
            mean_pbest = np.mean(self.Swarm.pbest)
            var_pbest = np.var(self.Swarm.pbest)

            data = [idx, gbest, mean_pbest, var_pbest]
            assert len(data) == len(self._headers)

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
