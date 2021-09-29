"""Implementation of the Surrogate class for the Particle Swarm Optimization (PSO)."""

import numpy as np
import plotly.graph_objects as go
import GPyOpt
from GPyOpt.models.gpmodel import GPModel, GPModel_MCMC

from psoas.utils import TimeDataBuffer, ValueDataBuffer

class Surrogate():
    """Surrogate class implementation.

    The class includes all functions related to the surrogate. Basically, a model is
    generated on the basis of the sampled points and their function values, on which
    a proposed optimum is then determined. To obtain a more narrow basis of data points 
    (e.g. for a local fit), a ringbuffer can be used.
    

    Attributes:
        surrogate_options: Dict containing options that belong to the surrogate
        sm: A surrogate model instance which is used to approximate the function based
            on the given data points
        dim: The dimension of the search-space
        n_particles: The number of particles in the swarm
        positions: Default memory for the positions which have been encountered so far.
        f_vals: Default memory for the function values which have been encountered so far.
        surrogate_memory: A instance of the ring buffer for the surrogate
            (Only if a buffer is selected)
    """

    def __init__(self, init_position, init_f_vals, surrogate_options=None):
        """ Creates and initializes a surrogate class instance.

        Args:
            init_position: Initial positions for the construction of the surrogate
                model as array of shape (n, dim)
            init_f_vals: Initial function values for the construction of the surrogate
                model as array of shape (n,)
            surrogate_options: Dict with which the method of the surrogate is selected
        """
        self.surrogate_options = surrogate_options
        if type(self.surrogate_options['surrogate_type']) == str:
            if (self.surrogate_options['surrogate_type'] == 'GP' or
                self.surrogate_options['surrogate_type'] == 'GP_MPI'):
                self.sm = GPModel(exact_feval = True, verbose=False)
            elif self.surrogate_options['surrogate_type'] == 'GP_MCMC':
                self.sm = GPModel_MCMC(exact_feval = True, verbose=False)
            else:
                raise ValueError(
                    "Expected GP, GP_MPI or GP_MCMC as parameter for initialization" \
                    f" of the model. Got {surrogate_options['surrogate_type']}.")
        else:
            raise ValueError(
                "Expected string as parameter. Got a"\
                f" {type(self.surrogate_options['surrogate_type'])} type.")

        self.dim = init_position.shape[1]
        self.n_particles = init_position.shape[0]

        self.positions = init_position.copy()
        self.f_vals = init_f_vals.copy()

        if self.surrogate_options['use_buffer']:
            if self.surrogate_options['buffer_type'] == 'time':
                self.surrogate_memory = TimeDataBuffer(
                    self.dim, self.n_particles, self.surrogate_options['n_slots'])

            elif self.surrogate_options['buffer_type'] == 'value':
                self.surrogate_memory = ValueDataBuffer(
                    self.dim, self.n_particles * self.surrogate_options['n_slots'])
            self.surrogate_memory.store(init_position, init_f_vals[:, None])

        self.sm.updateModel(init_position, init_f_vals[:, None], None, None)

    def fit_model(self, curr_positions, curr_f_vals):
        """Function that fits the surrogate model.

        This implementation fits a surrogate model based on the previous positions and
        function values as also the current data points. The latter should help to ensure
        that the surrogate is very accurate in the neighborhood of the currently
        sampled positions.

        Args:
            curr_positions: The positions of the current swarm particles as array
                with shape (n_particles, dim)
            curr_f_vals: The function values of the current swarm particles as array
                with shape (n_particles,)
        """
        input_positions = np.concatenate((self.positions, curr_positions), axis=0)
        
        input_f_vals = np.concatenate((self.f_vals, curr_f_vals))

        self.sm.updateModel(input_positions, input_f_vals[:, None], None, None)

    def update_data(self, curr_positions, curr_f_vals, do_filtering, mean=0, std=1, rho=1.15):
        """Updates the data of the surrogate model.

        Default data memory is updated by this function. This can be done with
        filtering to check, if a data point is informative. Otherwise, all data
        points are used for the surrogate model.

        Args:
            curr_positions: The positions of the current swarm particles as array
                with shape (n_particles, dim)
            curr_f_vals: The function values of the current swarm particles as array
                with shape (n_particles,)
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
        """Fitting the surrogate model based on the data points of the buffer.

        The data basis for the surrogate model is fetched from the buffer instance.
        """
        input_positions, input_f_vals = self.surrogate_memory.fetch()
        input_positions, idx = np.unique(input_positions, return_index=True, axis=0)
        input_f_vals = input_f_vals[idx]

        self.sm.updateModel(input_positions, input_f_vals, None, None)

    def update_data_buffer(self, curr_positions, curr_f_vals):
        """Function for updating the data buffer.

        Store the current positions and corresponding function values of the swarm
        in the buffer.

        Args:
            curr_positions: The positions of the current swarm particles as array
                with shape (n_particles, dim)
            curr_f_vals: The function values of the current swarm particles as array
                with shape (n_particles,)
        """

        self.surrogate_memory.store(curr_positions, curr_f_vals[:, None])

    def predict(self, positions):
        """Predicts the function value for a given positions.

        Args:
            positions: The positions from which we want to predict the function value

        Returns:
            predicted_mean: Predicted mean of the given position
            predicted_std: Predicted standard deviation of the given position
        """

        assert positions.shape[1] == (self.dim), (
            "The dimension of the positions does \
            not match with the dimension of the model. " \
            f"Expect dimension {self.dim}, got {positions.shape[1]}")

        predicted_mean, predicted_std = self.sm.predict(positions)
        return predicted_mean, predicted_std

    def update_surrogate(self, positions, f_vals):
        """Managing the updates of the surrogate model.

        This function handles the update of the surrogate. A new surrogate model
        is fitted on the basis of positions and functions values either withe the default
        memory or a buffer.

        Args:
            positions: An array of positions with which a new surrogate model is to
                be fitted
            f_vals: An array of function values with which a new surrogate model is to
                be fitted
        """
        if self.surrogate_options['use_buffer']:
            self.update_data_buffer(positions, f_vals)
            self.fit_model_buffer()
        else:
            mean, std = self.predict(positions)

            self.fit_model(positions, f_vals)
            self.update_data(positions, f_vals, True, mean, std)

    def get_proposition_point(self, constr):
        """Returns a proposed point.

        Given the surrogate model and the constraints of the search-space a acquisition
        function is constructed. To obtain a proposed point, a local minimum of the
        acquisition function must be determined. This is done via an optimizer.

        Args:
            constr: Constraints of the search space with shape (dim, 2)

        Returns:
            pos_proposition: Proposed position by the acquisition function (1, dim)
            f_val_proposition: Proposed function value by the acquisition function
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

        pos_proposition, f_val_proposition = acquisition.optimize()
        return pos_proposition, f_val_proposition[0][0]

    def use_standard_m_proposition(self, Swarm):
        """Applies the standard m algorthim which iteratively predicts m candidates using
        the surrogate and replaces the worst m swarm elements with these points.

        Args:
            Swarm: Instance of the Swarm 

        Returns:
            worst_indices: Indices of the worst m particles
            other_indices: Indices of the particles excluding the worst m particles
        """
        m = self.surrogate_options['m']

        # sort the pbest of the swarm and get the worst m particles
        worst_indices = np.argsort(Swarm.pbest)[-m:][::-1]
        other_indices = np.argsort(Swarm.pbest)[:-m][::-1]

        m_proposition_points = []
        m_proposition_values = []

        for i in range(m):

            position_proposition, f_val_proposition = self.get_proposition_point(Swarm.constr)
            proposition_point = position_proposition[0]

            f_val = Swarm.func(proposition_point[None,:])

            m_proposition_points.append(proposition_point)
            m_proposition_values.append(f_val)

            Swarm.positions[worst_indices[i]] = proposition_point
            Swarm.f_values[worst_indices[i]] = f_val
            
            if self.surrogate_options['use_buffer']:
                input_positions, input_f_vals = self.surrogate_memory.fetch()

                tmp_positions = np.vstack((input_positions, m_proposition_points))
                tmp_f_vals = np.vstack((input_f_vals, m_proposition_values))

                tmp_positions, idx = np.unique(tmp_positions, return_index=True, axis=0)
                tmp_f_vals = tmp_f_vals[idx]

                self.sm.updateModel(tmp_positions, tmp_f_vals, None, None)

            else:
                self.update_surrogate(self.Swarm.positions[worst_indices[i]][None, :], 
                                      np.atleast_1d(self.Swarm.f_values[worst_indices[i]]))

        Swarm.enforce_constraints(check_position=True, check_velocity=False)
        return worst_indices, other_indices

    def plot_surrogate(self):
        """Plotting the surrogate model.

        For the 2-dimensional case, a 3-dimensional plot is created, which contains the
        predicted surrogate model with the current data points. In addition, the variance
        for each point is shown in another 3-dimensional plot.
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
        fig.add_trace(go.Scatter3d(x=self.positions[:,0], y=self.positions[:,1], z=self.f_vals, mode='markers'))
        fig.show()

        fig = go.Figure(data=[go.Surface(x=axis, y=axis, z=predict_var_reshaped)])
        fig.show()
        print(80 * "*")
