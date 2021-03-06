"""Implementation of the evaluation framework for the Particle Swarm Optimizer. """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm

from psoas.optimizer import Optimizer


class Evaluation():
    """Base class for the evaluation classes.

    Implements basic functions for the other evaluation classes.
    """
    def __init__(self):
        pass

    def _create_dataframe(self, keys, length):
        """Creates a pandas data frame with the given keys and length."""

        data_dict = {}
        for key in keys:
            data_dict[key] = np.zeros(length)
        self.df = pd.DataFrame.from_dict(data_dict)

    def _optimize_function(self, func, n_particles, dim, constr, max_iter, max_func_evals, options=None):
        """Creates a PSOAS optimizer instance and uses it to optimize the given function."""

        opt = Optimizer(func, n_particles, dim, constr, max_iter, max_func_evals, **options)
        
        if self.eval_convergence_plot:
            opt.options['eval_convergence_plot'] = True

        return opt.optimize()

    def print_tables(self):
        try:
            display(self.df)
        except:
            print(self.df)
    
    @staticmethod
    def plot_percentiles(data, key):
        """Plots a percentiles plot for the given data and key."""

        y_data = np.fromiter(data[key].values(), dtype=np.float)
        x_data = np.linspace(0, 100, 1000)
        percentile = np.percentile(y_data, x_data)
        plt.plot(x_data, percentile)
        plt.xlabel('percentage')
        plt.ylabel(f'percentile for {key}')
        plt.show()
    
    @staticmethod
    def plot_histogram(data, key):
        """Plots a histogram plot for the given data and key."""

        y_data = np.fromiter(data[key].values(), dtype=np.float)
        plt.hist(y_data, bins=100)
        plt.xlabel('bins')
        plt.ylabel(f'histogram for {key}')
        plt.show()


class EvaluationSingle(Evaluation):
    """Evaluates a given function by optimizing it a set number of times and calculating
    statistical information from the runs.
    """

    def __init__(self, func, constr, ground_truth=None, opt_value=None):
        """Initializes the single function evaluation.
        
        Args:
            func: The function to be optimized
            constr: The constraints of the search-space with shape (dim, 2)
            ground_truth: (optional) The position of the optimum
            opt_value: (optional) The function value of the optimal position.
                Is necessary for some statistical information
        """
        self.func = func
        self.constr = constr
        if ground_truth is not None:
            self.ground_truth = ground_truth
        if opt_value is not None:
            self.opt_value = opt_value

    def evaluate_function(self, n_particles, dim, max_iter, max_func_evals, options, n_runs, 
                          eval_convergence_plot=False, disable_tqdm=False):
        """Repeatedly optimizes the function and stores the results for further analysis.

        Args:
            n_particles: The number of particles in the swarm
            dim: The dimension of the search-space
            max_iter: A integer value which determines the maximum amount of iterations
                in an optimization call
            max_func_evals: The maximum amount of function evaluations in an optimization call
            options: A dict containing all options which can be set for the optimization
            n_runs: The number of times the optimization is repeated
            eval_convergence_plots: A boolean which determines if the convergence data should be
                averaged and stored throughout the runs
            disable_tqdm: A boolean which determines if tqdm should be disabled (True: no tqdm, 
                False: tqdm is used)
        """
        keys = ['n_iter', 'n_fun_evals', 'term_flag', 'func_opt', 'mean_pbest', 'var_pbest']
        counter = 0
        if hasattr(self, 'ground_truth'):
            keys.append('dist_gt')
        if hasattr(self, 'opt_value'):
            keys.insert(3, 'diff_opt_value')
        self._create_dataframe(keys, n_runs)

        self.eval_convergence_plot = eval_convergence_plot

        for i in tqdm(range(n_runs), disable=disable_tqdm):
            # Here, a special error that occurred only on one server is caught during evaluation.
            # This problem could not be investigated further within the scope of our project work.
            # Seams to be a problem with GPyOpt and this specific server.
            error = True
            while error:
                try:
                    res = self._optimize_function(self.func, n_particles, dim, self.constr, max_iter, max_func_evals, options)
                    error = False
                except np.linalg.LinAlgError:
                    counter += 1
                    error = True
                    if counter > 200:
                        raise RuntimeError(f'Tried more than {counter}')
            self.df.loc[i, 'n_iter'] = res['iter']
            self.df.loc[i, 'n_fun_evals'] = res['n_fun_evals']
            self.df.loc[i, 'term_flag'] = res['term_flag']
            self.df.loc[i, 'func_opt'] = res['func_opt']
            self.df.loc[i, 'mean_pbest'] = res['mean_pbest']
            self.df.loc[i, 'var_pbest'] = res['var_pbest']

            # prepare averaged convergence plots
            if self.eval_convergence_plot:
                if not hasattr(self, 'gbests'):
                    self.counting_run = 1
                    self.gbests = np.array(res['gbest_list'])
                    self.mean_pbests = np.array(res['mean_pbest_list'])
                    self.var_pbests = np.array(res['var_pbest_list'])
                    self.n_fun_evals = np.array(res['n_fun_eval_list'])

                else:
                    self.gbests = (self.gbests * self.counting_run + np.array(res['gbest_list']))  / (self.counting_run + 1)
                    self.mean_pbests = (self.mean_pbests * self.counting_run + np.array(res['mean_pbest_list']))  / (self.counting_run + 1)
                    self.var_pbests = (self.var_pbests * self.counting_run + np.array(res['var_pbest_list']))  / (self.counting_run + 1)
                    self.n_fun_evals = (self.n_fun_evals * self.counting_run + np.array(res['n_fun_eval_list'])) / (self.counting_run + 1)
                    self.counting_run += 1

            if 'dist_gt' in self.df.keys():
                assert self.ground_truth.shape == res['x_opt'].shape
                self.df.loc[i, 'dist_gt'] = np.linalg.norm(res['x_opt']- self.ground_truth)
            if 'diff_opt_value' in self.df.keys():
                self.df.loc[i, 'diff_opt_value'] = res['func_opt'] - self.opt_value
        if counter > 0:
            print(f'Repeats for function: {counter}')

    def plot_convergence(self):
        """Plots the gbest averaged over all runs and the mean and standard deviations of
        the personal bests on the ordinate of two subplots respectively. They are plotted
        in regard to the function evaluations which are shown on the abscissa.
        """
        assert self.eval_convergence_plot, 'The data for the convergence plots has to be prepared during the optimizations.'
        
        fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(12,12))

        axs[0].plot(self.n_fun_evals, self.gbests, color='orange')
        axs[0].set_ylabel('averaged gbest fval')

        axs[1].plot(self.n_fun_evals, self.mean_pbests, color='tab:blue')
        axs[1].fill_between(self.n_fun_evals, self.mean_pbests - np.sqrt(self.var_pbests), 
                            self.mean_pbests + np.sqrt(self.var_pbests), color='tab:blue', alpha=0.2)

        axs[1].set_ylabel('averaged mean +- std pbest fval')
        axs[1].set_xlabel('function evaluations')

        fig.show()
        
    def get_statistical_information(self):
        """Averages and sums up the data gathered throughout all of the runs."""
        
        mean_iters = np.mean(self.df['n_iter'])
        mean_fun_evals = np.mean(self.df['n_fun_evals'])
        mean = np.mean(self.df['func_opt'])
        var = np.var(self.df['func_opt'])
        min = np.min(self.df['func_opt'])
        max = np.max(self.df['func_opt'])
        median = np.median(self.df['func_opt'])
        std = np.std(self.df['func_opt'])
        
        if hasattr(self, 'opt_value'):
            diff = self.df['func_opt'] - self.opt_value

            mean_diff = np.mean(diff)
            var_diff = np.var(diff)
            min_diff = np.min(diff)
            max_diff = np.max(diff)

            stats_dict = {'min': min,'median': median, 'mean': mean, 'max': max, 'std': std,
                        'var': var, 'mean_iters': mean_iters, 'mean_fun_evals': mean_fun_evals,
                        'min_diff': min_diff, 'mean_diff': mean_diff, 'max_diff': max_diff,   
                        'var_diff': var_diff}
        
        else:
            stats_dict = {'min': min,'median': median, 'mean': mean, 'max': max, 'std': std,
                        'var': var, 'mean_iters': mean_iters, 'mean_fun_evals': mean_fun_evals}

        for key in stats_dict.keys():
            stats_dict[key] = np.round(stats_dict[key], 5)
        return stats_dict

    def plot_percentiles(self, key):
        super().plot_percentiles(self.df.to_dict(), key)

    def plot_histogram(self, key):
        super().plot_histogram(self.df.to_dict(), key)


class EvaluationFunctionSet(Evaluation):

    def evaluate_functions(self, bench, n_particles, dim, max_iter, max_func_evals, options, n_runs):
        """Goes through all the functions of the CEC2013 benchmark, repeatedly optimizes them and gathers the results.
        The results for each function are averaged and summed up along the runs.

        Args:
            bench: An instance of the CEC2013 benchmark class
            n_particles: The number of particles in the swarm
            dim: The dimension of the search-space
            max_iter: A integer value which determines the maximum amount of iterations
                in an optimization call
            max_func_evals: The maximum amount of function evaluations in an optimization call
            options: A dict containing all options which can be set for the optimization
            n_runs: The number of times the optimization is repeated
        """
        self.n_particles = n_particles
        self.dim = dim
        self.max_iter = max_iter
        self.max_func_evals = max_func_evals

        keys = ['min', 'median', 'mean', 'max', 'std', 'var', 'mean_iters', 'mean_fun_evals', 'min_diff', 'mean_diff', 'max_diff', 'var_diff']
        self.df = pd.DataFrame(columns=keys, index=range(0,28))

        self.results = {}

        for idx in tqdm(range(1,29)):
            func = bench.get_function(idx)
            info = bench.get_info(idx, dim)
            constraints = np.ones((dim, 2))*np.array([info['lower'], info['upper']])

            eval_single = EvaluationSingle(func, constraints, opt_value=info['best'])
            eval_single.evaluate_function(self.n_particles, self.dim, self.max_iter, self.max_func_evals, options, 
                                          n_runs, disable_tqdm=True)
            self.results[str(idx)] = eval_single.df.to_dict()

            func_stat_data = eval_single.get_statistical_information()
            self.df.iloc[idx-1] = pd.Series(func_stat_data)
        
        self.results['summary'] = self.df.to_dict()

    def plot_percentiles(self, func_idx, key):
        super().plot_percentiles(self.results[str(func_idx)], key)

    def plot_histogram(self, func_idx, key):
        super().plot_histogram(self.results[str(func_idx)], key)
