"""
TODO: Docstring
"""

import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import GPyOpt
from GPyOpt.models.gpmodel import GPModel

class Surrogate():
    """
    Docstring: TODO
    """

    def __init__(self, init_position, init_f_val, surrogate_options=None):
        """
        Docstring: TODO
        """
        if type(surrogate_options['surrogate_type']) == str:
            if surrogate_options['surrogate_type'] == 'GP':
                self.sm = GPModel(exact_feval = True, verbose=False)
            else:
                raise ValueError(f"Expected GP as parameter for initialization of the model. Got {surrogate_options['surrogate_type']}.")
        else:
            raise ValueError(f"Expected string as parameter. Got a {type(surrogate_options['surrogate_type'])} type.")

        self.positions = init_position.copy()
        self.f_val = init_f_val.copy()

        self.dim = init_position.shape[1]
        self.n_particles = init_position.shape[0]

    def update_data(self, curr_position, curr_f_val):
        """
        Docstring: TODO
        """
        self.positions = np.concatenate((self.positions, curr_position), axis=0)
        self.f_val = np.concatenate((self.f_val, curr_f_val))

        self.positions, idx = np.unique(self.positions, return_index=True, axis=0)
        self.f_val = self.f_val[idx]

    def fit_model(self):
        """
        Docstring: TODO
        """
        self.sm.updateModel(self.positions, self.f_val[:, None], None, None)

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
        acquisition = GPyOpt.acquisitions.EI.AcquisitionEI(self.sm, space, acquisition_optimizer)
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
        fig.add_trace(go.Scatter3d(x=self.positions[:,0], y=self.positions[:,1], z=self.f_val, mode='markers'))
        fig.show()
        print(80 * "*")
