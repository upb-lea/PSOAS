"""Implementation of the evaluation framework for the Particle Swarm Optimizer. """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm

from psoas.optimizer import Optimizer


class Evaluation():
    """
    TODO: docstring
    """
    def __init__(self):
        pass

    def _create_dataframe(self, keys, height):
        data_dict = {}
        for key in keys:
            data_dict[key] = np.zeros(height)
        self.df = pd.DataFrame.from_dict(data_dict)

    def _optimize_function(self, func, n_particles, dim, constr, max_iter, max_func_evals, options=None):
        opt = Optimizer(func, n_particles, dim, constr, max_iter, max_func_evals, **options)
        return opt.optimize()

    def print_tables(self):
        try:
            display(self.df)
        except:
            print(self.df)
    
    @staticmethod
    def plot_percentiles(data, key):
        y_data = np.fromiter(data[key].values(), dtype=np.float)
        x_data = np.linspace(0, 100, 1000)
        percentile = np.percentile(y_data, x_data)
        plt.plot(x_data, percentile)
        plt.xlabel('percentage')
        plt.ylabel(f'percentile for {key}')
        plt.show()
    
    @staticmethod
    def plot_histogram(data, key):
        y_data = np.fromiter(data[key].values(), dtype=np.float)
        plt.hist(y_data, bins=100)
        plt.xlabel('bins')
        plt.ylabel(f'histogram for {key}')
        plt.show()


class EvaluationSingle(Evaluation):
    """
    TODO: docstring

    One function, multiple runs, statistical information 
    """

    def __init__(self, func, constr, ground_truth=None, opt_value=None):
        self.func = func
        self.constr = constr
        if ground_truth is not None:
            self.ground_truth = ground_truth
        if opt_value is not None:
            self.opt_value = opt_value

    def evaluate_function(self, n_particles, dim, max_iter, max_func_evals, options, n_runs, disable_tqdm=False):
        keys = ['n_iter', 'n_fun_evals', 'term_flag', 'func_opt', 'mean_pbest', 'var_pbest']
        counter = 0
        if hasattr(self, 'ground_truth'):
            keys.append('dist_gt')
        if hasattr(self, 'opt_value'):
            keys.insert(3, 'diff_opt_value')
        self._create_dataframe(keys, n_runs)
        for i in tqdm(range(n_runs), disable=disable_tqdm):
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
            if 'dist_gt' in self.df.keys():
                assert self.ground_truth.shape == res['x_opt'].shape
                self.df.loc[i, 'dist_gt'] = np.linalg.norm(res['x_opt']- self.ground_truth)
            if 'diff_opt_value' in self.df.keys():
                self.df.loc[i, 'diff_opt_value'] = res['func_opt'] - self.opt_value
        if counter > 0:
            print(f'Repeats for function: {counter}')
        
    def get_statistical_information(self):
        mean_iters = np.mean(self.df['n_iter'])
        mean_fun_evals = np.mean(self.df['n_fun_evals'])
        mean = np.mean(self.df['func_opt'])
        var = np.var(self.df['func_opt'])
        min = np.min(self.df['func_opt'])
        max = np.max(self.df['func_opt'])
        median = np.median(self.df['func_opt'])
        std = np.std(self.df['func_opt'])

        diff = self.df['func_opt'] - self.opt_value

        mean_diff = np.mean(diff)
        var_diff = np.var(diff)
        min_diff = np.min(diff)
        max_diff = np.max(diff)
        
    
        stats_dict = {'min': min,'median': median, 'mean': mean, 'max': max, 'std': std,
                      'var': var, 'mean_iters': mean_iters, 'mean_fun_evals': mean_fun_evals,
                      'min_diff': min_diff, 'mean_diff': mean_diff, 'max_diff': max_diff,   
                      'var_diff': var_diff}
    
        for key in stats_dict.keys():
            stats_dict[key] = np.round(stats_dict[key], 5)
        return stats_dict

    def plot_percentiles(self, key):
        super().plot_percentiles(self.df.to_dict(), key)

    def plot_histogram(self, key):
        super().plot_histogram(self.df.to_dict(), key)


class EvaluationHyperparameters(Evaluation):    
    """
    TODO: docstring

    One function, multiple runs, different hyperparameters
    """

    def __init__(self):
        pass

    def evaluate_function():
        raise NotImplementedError


class EvaluationFunctionSet(Evaluation):
    """
    TODO: docstring

    Multiple functions, (multiple? runs)
    """

    def evaluate_functions(self, bench, n_particles, dim, max_iter, max_func_evals, options, n_runs):
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
