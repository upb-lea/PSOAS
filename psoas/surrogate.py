"""
TODO: Docstring
"""

import numpy as np
import plotly.graph_objects as go
import GPyOpt
from GPyOpt.models.gpmodel import GPModel, GPModel_MCMC

from psoas.utils import TimeDataBuffer, ValueDataBuffer

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

        self.dim = init_position.shape[1]
        self.n_particles = init_position.shape[0]

        self.positions = init_position.copy()
        self.f_vals = init_f_vals.copy()

        if self.surrogate_options['use_buffer']:
            if self.surrogate_options['buffer_type'] == 'time':
                self.surrogate_memory = TimeDataBuffer(self.dim, self.n_particles, self.surrogate_options['n_slots'])

            elif self.surrogate_options['buffer_type'] == 'value':
                self.surrogate_memory = ValueDataBuffer(self.dim, self.n_particles * self.surrogate_options['n_slots'])
            self.surrogate_memory.store(init_position, init_f_vals[:, None])

        self.sm.updateModel(init_position, init_f_vals[:, None], None, None)

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

            if curr_positions.shape[0] == 1:
                if not helpful_points:
                    return
            else:
                curr_positions = curr_positions[helpful_points]
                curr_f_vals = curr_f_vals[helpful_points]

        self.positions = np.concatenate((self.positions, curr_positions), axis=0)
        self.f_vals = np.concatenate((self.f_vals, curr_f_vals))

        self.positions, idx = np.unique(self.positions, return_index=True, axis=0)
        self.f_vals = self.f_vals[idx]
    
    def fit_model_buffer(self):
        """
        TODO:Docstring
        """
        input_positions, input_f_vals = self.surrogate_memory.fetch()
        input_positions, idx = np.unique(input_positions, return_index=True, axis=0)
        input_f_vals = input_f_vals[idx]

        self.sm.updateModel(input_positions, input_f_vals, None, None)

    def update_data_buffer(self, curr_positions, curr_f_vals):
        """
        TODO: Docstring
        """
        self.surrogate_memory.store(curr_positions, curr_f_vals[:, None])

    def predict(self, positions):
        """
        TODO: docsting
        """
        assert point.shape[1] == (self.dim), f'The dimension of the point does not match with the dimension of the model. Expect dimension {self.dim}, got {point.shape[0]}'
        predicted_mean, predicted_std = self.sm.predict(positions)
        return predicted_mean, predicted_std

    def update_surrogate(self, positions, f_values):
        """
        This function handles the update of the surrogate by estimating mean and std 
        for the incoming PSO data points, then creating a new model based on the surrogate's 
        data points and the PSO's current data points, and then updating the data while checking
        if the new point have an influcence on the surrogate model.
        """
        if self.surrogate_options['use_buffer']:
            self.update_data_buffer(positions, f_values)
            self.fit_model_buffer()
        else:
            mean, std = self.predict(positions)

            self.fit_model(positions, f_values)
            self.update_data(positions, f_values, True, mean, std)

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

        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space, optimizer='lbfgs')

        if self.surrogate_options['surrogate_type'] == 'GP':
            acquisition = GPyOpt.acquisitions.AcquisitionEI(self.sm, space, acquisition_optimizer)
        elif self.surrogate_options['surrogate_type'] == 'GP_MPI':
            acquisition = GPyOpt.acquisitions.AcquisitionMPI(self.sm, space, acquisition_optimizer)
        elif self.surrogate_options['surrogate_type'] == 'GP_MCMC':
            acquisition = GPyOpt.acquisitions.AcquisitionEI_MCMC(self.sm, space, acquisition_optimizer)

        pos_prediction, f_val_prediction = acquisition.optimize()
        return pos_prediction, f_val_prediction

    def use_standard_m_prediction(self, Swarm):
        """Applies the standard m algortihm which iteratively predicts m candidates using
        the surrogate and replaces the worst m swarm elements with these points.
        """
        m = self.surrogate_options['m']

        worst_indices = np.argsort(Swarm.pbest)[-m:][::-1]
        other_indices = np.argsort(Swarm.pbest)[:-m][::-1]

        m_prediction_points = []
        m_prediction_values = []

        for i in range(m):

            position_prediction, f_val_prediction = self.get_prediction_point(Swarm.constr)
            prediction_point = position_prediction[0]

            f_val = Swarm.func(prediction_point[None,:])

            m_prediction_points.append(prediction_point)
            m_prediction_values.append(f_val)

            Swarm.positions[worst_indices[i]] = prediction_point
            Swarm.f_values[worst_indices[i]] = f_val
            
            if self.surrogate_options['use_buffer']:
                input_positions, input_f_vals = self.surrogate_memory.fetch()

                tmp_positions = np.vstack((input_positions, m_prediction_points))
                tmp_f_vals = np.vstack((input_f_vals, m_prediction_values))

                tmp_positions, idx = np.unique(tmp_positions, return_index=True, axis=0)
                tmp_f_vals = tmp_f_vals[idx]

                self.sm.updateModel(tmp_positions, tmp_f_vals, None, None)

            else:
                self.update_surrogate(self.Swarm.positions[worst_indices[i]][None, :], 
                                                    np.atleast_1d(self.Swarm.f_values[worst_indices[i]]))

        Swarm.enforce_constraints(check_position=True, check_velocity=False)
        return worst_indices, other_indices

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
