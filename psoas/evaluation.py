"""
Implementation of the evaluation framwork for the Particle Swarm Optimizer. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm

from psoas.optimizer import Optimizer

class Evaluation():
    """
    TODO: docstring
    """

    def __init__(self, function_list, constr_list):
        self.function_list = function_list
        self.constr_list = constr_list

    def _evaluate_function(self, func, constr):
        opt = Optimizer(func, self.n_particles, self.dim, constr, self.max_iter)
        return opt.optimize()

    def evaluate_functions(self, n_particles, dim, max_iter):
        self.n_particles = n_particles
        self.dim = dim
        self.max_iter = max_iter

        result_df = pd.DataFrame()

        for idx, func in tqdm(enumerate(self.function_list)):
            res = self._evaluate_functions(func, self.constr_list[idx, :])

    def print_tables():
        raise NotImplementedError