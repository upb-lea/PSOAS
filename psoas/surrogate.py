"""
TODO: Docstring
"""

import numpy as np
from numpy.ma.extras import vander
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import GPyOpt
from GPyOpt.models.gpmodel import GPModel, GPModel_MCMC

class Surrogate():
    """
    Docstring: TODO
    """

    def __init__(self, init_position, init_f_vals, surrogate_options=None):
        """
        Docstring: TODO
        """
        self.surrogate_options = surrogate_options
        if type(self.surrogate_options['surrogate_type']) == str:
            if self.surrogate_options['surrogate_type'] == 'GP':
                self.sm = GPModel(exact_feval = True, verbose=False)
            elif self.surrogate_options['surrogate_type'] == 'GP_MCMC':
                self.sm = GPModel_MCMC(exact_feval = True, verbose=False)
            else:
                raise ValueError(f"Expected GP or GP_MCMC as parameter for initialization of the model. Got {surrogate_options['surrogate_type']}.")
        else:
            raise ValueError(f"Expected string as parameter. Got a {type(self.surrogate_options['surrogate_type'])} type.")

        self.positions = init_position.copy()
        self.f_vals = init_f_vals.copy()

        self.dim = init_position.shape[1]
        self.n_particles = init_position.shape[0]

    def fit_model(self, curr_positions, curr_f_vals):
        """
        Docstring: TODO
        """
        input_positions = np.concatenate((self.positions, curr_positions), axis=0)
        input_f_vals = np.concatenate((self.f_vals, curr_f_vals))

        self.sm.updateModel(input_positions, input_f_vals[:, None], None, None)


    def update_data(self, curr_positions, curr_f_vals, rho=1.5):
        """
        Docstring: TODO
        """
        # discard any points that hold little information
        # see Jakubik2021
        mean, std = self.sm.predict(curr_positions)
        var = std**2
        lower_bound = mean - rho * var
        upper_bound = mean + rho * var

        lower_bound = np.reshape(lower_bound, (lower_bound.shape[0],))
        upper_bound = np.reshape(upper_bound, (upper_bound.shape[0],))

        lower_bool = np.greater(curr_f_vals, lower_bound)
        upper_bool = np.less(curr_f_vals, upper_bound)

        helpful_points = ~np.logical_and(lower_bool, upper_bool)
        curr_positions = curr_positions[helpful_points]
        curr_f_vals = curr_f_vals[helpful_points]

        self.positions = np.concatenate((self.positions, curr_positions), axis=0)
        self.f_vals = np.concatenate((self.f_vals, curr_f_vals))

        self.positions, idx = np.unique(self.positions, return_index=True, axis=0)
        self.f_vals = self.f_vals[idx]

    def predict(self, point):
        """
        Docstring: TODO
        """
        assert point.shape[1] == (self.dim), f'The dimension of the point does not match with the dimension of the model. Expect dimension {self.dim}, got {point.shape[0]}'
        predict_mean, predict_std = self.sm.predict(point)
        return predict_mean, predict_std

    def get_prediction_point(self, constr):
        mixed_domain = []
        for c in constr:
            dim_dict = {'name': 'var1', 'type': 'continuous', 'domain': tuple(c)}
            mixed_domain.append(dim_dict)
        space = GPyOpt.Design_space(mixed_domain)

        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)

        if self.surrogate_options['surrogate_type'] == 'GP':
            acquisition = GPyOpt.acquisitions.MPI.AcquisitionMPI(self.sm, space, acquisition_optimizer, jitter=0)
        elif self.surrogate_options['surrogate_type'] == 'GP_MCMC':
            acquisition = GPyOpt.acquisitions.AcquisitionEI_MCMC(self.sm, space, acquisition_optimizer)

        prediction = acquisition.optimize()
        return prediction

    def plotter_3d(self):
        """
        Docstring: TODO
        """
        assert self.dim == 2, f'Expect dimension to be 2! Got {self.dim}.'
        num = 100
        axis = np.linspace(-100, 100, num)
        x, y = np.meshgrid(axis, axis)

        x = x.flatten()
        y = y.flatten()
        flat_grid = np.array([x, y]).T

        predict_val, _ = self.predict(flat_grid)

        predict_val_reshaped = np.reshape(predict_val, (num, num))

        print(80 * "*")

        fig = go.Figure(data=[go.Surface(x=axis, y=axis, z=predict_val_reshaped)])
        fig.add_trace(go.Scatter3d(x=self.positions[:,0], y=self.positions[:,1], z=self.f_vals, mode='markers'))
        fig.show()
        print(80 * "*")
