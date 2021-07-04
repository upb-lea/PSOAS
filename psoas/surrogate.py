"""
TODO: Docstring
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from smt.surrogate_models import RBF
from smt.surrogate_models import KRG

class Surrogate():
    """
    Docstring: TODO
    """

    def __init__(self, init_position, init_f_val, surrogate_options=None):
        """
        Docstring: TODO
        """
        if type(surrogate_options['surrogate_type']) == str:
            if surrogate_options['surrogate_type'] == 'RBF':
                self.sm = RBF(d0=6)
            elif surrogate_options['surrogate_type'] == 'KRG':
                self.sm = KRG(theta0=[1e-2])
            else:
                raise ValueError(f"Expected RBF or KRG as parameter for initialization of the model. Got {surrogate_options['surrogate_type']}.")
        else:
            raise ValueError(f"Expected string as parameter. Got a {type(surrogate_options['surrogate_type'])} type.")
        
        self.sm.options['print_global'] = False
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
        self.sm.set_training_values(self.positions, self.f_val)
        self.sm.train()

    def predict(self, point):
        """
        Docstring: TODO
        """
        assert point.shape[1] == (self.dim), f'The dimension of the point does not match with the dimension of the model. Expect dimension {self.dim}, got {point.shape[0]}'
        predict_val = self.sm.predict_values(point)
        return predict_val

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

        predict_val = self.predict(flat_grid)

        predict_val_reshaped = np.reshape(predict_val, (num, num))

        print(80 * "*")

        fig = go.Figure(data=[go.Surface(x=axis, y=axis, z=predict_val_reshaped)])
        fig.add_trace(go.Scatter3d(x=self.positions[:,0], y=self.positions[:,1], z=self.f_val, mode='markers'))

        # fig.update_layout(autosize=False,
        #           width=500, height=500,
        #           margin=dict(l=65, r=50, b=65, t=90))
        fig.show()
        print(80 * "*")