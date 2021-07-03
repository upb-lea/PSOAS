"""
Implementation of the evaluation framwork for the Particle Swarm Optimizer. 
"""

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

    def _optimize_function(self, func, n_particles, dim, constr, max_iter, options=None):
        opt = Optimizer(func, n_particles, dim, constr, max_iter, options)
        return opt.optimize()

    def print_tables(self):
        try:
            display(self.df)
        except:
            print(self.df)


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

    def evaluate_function(self, n_particles, dim, max_iter, options, n_runs):
        keys = ['n_iter', 'term_flag', 'func_opt', 'mean_pbest', 'var_pbest']
        if hasattr(self, 'ground_truth'):
            keys.append('dist_gt')
        if hasattr(self, 'opt_value'):
            keys.insert(3, 'diff_opt_value')

        self._create_dataframe(keys, n_runs)
        for i in tqdm(range(n_runs)):
            res = self._optimize_function(self.func, n_particles, dim, self.constr, max_iter, options)
            self.df.loc[i, 'n_iter'] = res['iter']
            self.df.loc[i, 'term_flag'] = res['term_flag']
            self.df.loc[i, 'func_opt'] = res['func_opt']
            self.df.loc[i, 'mean_pbest'] = res['mean_pbest']
            self.df.loc[i, 'var_pbest'] = res['var_pbest']
            if 'dist_gt' in self.df.keys():
                assert self.ground_truth.shape == res['x_opt'].shape
                self.df.loc[i, 'dist_gt'] = np.linalg.norm(self.ground_truth - res['x_opt'])
            if 'diff_opt_value' in self.df.keys():
                self.df.loc[i, 'diff_opt_value'] = self.opt_value - res['func_opt']
 

class EvaluationHyperparameters(Evaluation):    
    """
    TODO: docstring

    One function, multiple runs, different hyperparameters
    """

    def __init__(self):
        pass


class EvaluationFunctionSet(Evaluation):
    """
    TODO: docstring

    Multiple functions, (multiple? runs)
    """

    def __init__(self, function_list, constr_list):
        self.function_list = function_list
        self.constr_list = constr_list

    def evaluate_functions(self, n_particles, dim, max_iter):
        self.n_particles = n_particles
        self.dim = dim
        self.max_iter = max_iter

        result_df = pd.DataFrame()

        for idx, func in tqdm(enumerate(self.function_list)):
            res = self._evaluate_functions(func, self.n_particles, self.dim, self.constr_list[idx, :], self.max_iter)
