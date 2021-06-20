"""
TODO: Docstring
"""

import numpy as np
import plotly.graph_objects as go

from smt.surrogate_models import RBF
from smt.surrogate_models import KRG

class Surrogate():

    def __init__(self, init_position, init_f_val, options=None):

        self.sm = KRG(theta0=[1e-2])
        self.sm.options['print_global'] = False
        self.positions = init_position
        self.f_val = init_f_val
        self.dim = init_position.shape[1]
        self.n_particels = init_position.shape[0]

    def update_data(self, curr_postion, curr_f_val):

        self.positions = np.concatenate((self.positions, curr_postion), axis=0)
        self.f_val = np.concatenate((self.f_val, curr_f_val))


    def fit_model(self):

        self.sm.set_training_values(self.positions, self.f_val)
        self.sm.train()


    def predict(self, point):
        assert point.shape[1] == (self.dim), f'The dimension of the point does not match with the dimension of the model. Expect dimension {self.dim}, got {point.shape[0]}'

        predict_val = self.sm.predict_values(point)

        return predict_val


    def plotter_2d(self):

        num = 100
        axis = np.linspace(-100, 100, num)
        x, y = np.meshgrid(axis, axis)

        x = x.flatten()
        y = y.flatten()
        flat_grid = np.array([x, y]).T

        predict_val = self.predict(flat_grid)

        predict_val_reshaped = np.reshape(predict_val, (num, num))

        print(80 * "*")
        fig = go.Figure(data=[go.Surface(z=predict_val_reshaped)])
        fig.show()
        print(80 * "*")