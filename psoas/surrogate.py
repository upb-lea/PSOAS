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
    Surrogate class implementation.

    The class includes all functions related to the surrogate. Basically, a model is generated
    on the basis of the sampled points and their function values, on which a predicted optimum is then determined. 
    """

    def __init__(self, init_position, init_f_vals, surrogate_options=None):
        """
        Creates and initializes a surrogate class instance.

        Args:
            init_position: The initial positions as array for the creation of a model (n, dim)
            init_f_vals: The initial function values for the creation of a model (n,)
            surrogate_options: Dict with which the method of the surrogate is selected
        """
        self.surrogate_options = surrogate_options
        if type(self.surrogate_options['surrogate_type']) == str:
            if self.surrogate_options['surrogate_type'] == 'GP' or self.surrogate_options['surrogate_type'] == 'GP_MPI':
                self.sm = GPModel(exact_feval = True, verbose=False)
            elif self.surrogate_options['surrogate_type'] == 'GP_MCMC':
                self.sm = GPModel_MCMC(exact_feval = True, verbose=False)
            else:
                raise ValueError(f"Expected GP or GP_MCMC as parameter for initialization of the model. Got {surrogate_options['surrogate_type']}.")
        else:
            raise ValueError(f"Expected string as parameter. Got a {type(self.surrogate_options['surrogate_type'])} type.")

        self.positions = init_position.copy()
        self.f_vals = init_f_vals.copy()
        self.sm.updateModel(init_position, init_f_vals[:, None], None, None)

        self.dim = init_position.shape[1]
        self.n_particles = init_position.shape[0]

    def fit_model(self, curr_positions, curr_f_vals):
        """
        This implementation creates a surrogate model based on the stored data points
        and the current data points provided by the PSO. This should help to ensure that
        the surrogate is very accurate in the neighborhood of the currently sampled points.

        Args:
            curr_positions: The position of the current swarm particles with shape (n, dim)
            curr_f_vals: The function values of the current swarm particles with shape (n,)
        """
        input_positions = np.concatenate((self.positions, curr_positions), axis=0)
        
        input_f_vals = np.concatenate((self.f_vals, curr_f_vals))

        self.sm.updateModel(input_positions, input_f_vals[:, None], None, None)

    def update_data(self, curr_positions, curr_f_vals, do_filtering, mean=0, std=1, rho=1.15):
        """
        The update of the surrogate data can be performed with a filtering,
        checking if a data point is informative. Otherwise, all data points are included in the model.

        Args:
            curr_positions: The position of the current swarm particles with shape (n, dim)
            curr_f_vals: The function values of the current swarm particles with shape (n,)
            do_filtering: Boolean for the filtering of the data points
        """
        if do_filtering:
            var = std**2
            helpful_points = var > np.ones_like(var) * self.dim**2

            helpful_points = np.squeeze(helpful_points)

            curr_positions = curr_positions[helpful_points]
            curr_f_vals = curr_f_vals[helpful_points]

        self.positions = np.concatenate((self.positions, curr_positions), axis=0)
        self.f_vals = np.concatenate((self.f_vals, curr_f_vals))

        self.positions, idx = np.unique(self.positions, return_index=True, axis=0)
        self.f_vals = self.f_vals[idx]

    def predict(self, point):
        """
        Calculates the function value for the predicted optimum on the model

        Args:
            point: Predicted optimum with shape (1, dim)
        """
        assert point.shape[1] == (self.dim), f'The dimension of the point does not match with the dimension of the model. Expect dimension {self.dim}, got {point.shape[0]}'
        predict_mean, predict_std = self.sm.predict(point)
        return predict_mean, predict_std

    def get_prediction_point(self, constr):
        """
        Searches the minimum of the surogate based on the constraints of the search space.

        Args:
            constr: Constraints of the search space with shape (???)

        Returns:
            prediction: prediction optimum with shape (1, dim)
        """
        mixed_domain = []
        for c in constr:
            dim_dict = {'name': 'var1', 'type': 'continuous', 'domain': tuple(c)}
            mixed_domain.append(dim_dict)
        space = GPyOpt.Design_space(mixed_domain)

        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)

        if self.surrogate_options['surrogate_type'] == 'GP':
            acquisition = GPyOpt.acquisitions.AcquisitionEI(self.sm, space, acquisition_optimizer, jitter=0)
        elif self.surrogate_options['surrogate_type'] == 'GP_MPI':
            acquisition = GPyOpt.acquisitions.AcquisitionMPI(self.sm, space, acquisition_optimizer, jitter=0)
        elif self.surrogate_options['surrogate_type'] == 'GP_MCMC':
            acquisition = GPyOpt.acquisitions.AcquisitionEI_MCMC(self.sm, space, acquisition_optimizer)

        prediction = acquisition.optimize()
        return prediction

    def plotter_3d(self):
        """
        For the 2-dimensional case, a 3-dimensional plot is created, which contains
        the surrogate with the current data points. In addition, the predicted variance
        is plotted in a further plot.
        """
        assert self.dim == 2, f'Expect dimension to be 2! Got {self.dim}.'
        num = 100
        axis = np.linspace(-100, 100, num)
        x, y = np.meshgrid(axis, axis)

        x = x.flatten()
        y = y.flatten()
        flat_grid = np.array([x, y]).T

        predict_mean, predict_var = self.predict(flat_grid)

        predict_mean_reshaped = np.reshape(predict_mean, (num, num))
        predict_var_reshaped = np.reshape(predict_var, (num, num))

        print(80 * "*")

        fig = go.Figure(data=[go.Surface(x=axis, y=axis, z=predict_mean_reshaped)])
        #fig.add_trace(go.Scatter3d(x=self.positions[:,0], y=self.positions[:,1], z=self.f_vals, mode='markers'))
        fig.show()

        fig = go.Figure(data=[go.Surface(x=axis, y=axis, z=predict_var_reshaped)])
        fig.show()
        print(80 * "*")
