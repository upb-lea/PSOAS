"""Implementation of the Surrogate class for the Particle Swarm Optimization (PSO)."""

import numpy as np
import plotly.graph_objects as go
import GPyOpt
from GPyOpt.models.gpmodel import GPModel, GPModel_MCMC

from psoas.utils import TimeDataBuffer, ValueDataBuffer

class Surrogate():
    """Surrogate class implementation.

    This class wraps the gaussian process model from GPyOpt. It manages the data with
    which the model is fitted, initiates the fit and wraps the acquisition function
    optimization, that the surrogate model uses to propose a point that potentially 
    improves on the optima found before. Since the computation times for the surrogate
    grow quickly with the amount of data points, a ringbuffer (either time- or value-
    based) can be used to reduce the data base instead of the standard memory where 
    essentially all unique data is kept.

    Attributes:
        surrogate_options: A Dict containing all options that belong to the surrogate
        sm: A GPyOpt gaussian process model which is used to approximate the function 
            based on the given data points
        dim: The dimension of the search-space
        n_particles: The number of particles in the swarm
        positions: Default memory for the positions which have been encountered so far.
        f_vals: Default memory for the function values which have been encountered so far.
        surrogate_memory: The ring buffer for the surrogate (Only if a buffer is selected)
    """

    def __init__(self, init_position, init_f_vals, surrogate_options=None):
        """ Creates and initializes a surrogate class instance.

        Args:
            init_position: Initial positions of shape (n, dim) of the data for the 
                construction of the surrogate model as array 
            init_f_vals: Initial function values for the construction of the surrogate
                model as array of shape (n,)
            surrogate_options: A Dict containing all options that belong to the surrogate
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

        This function fits a surrogate model based on the previous positions and
        function values in the surrogate memory as well as the current data points.
        The latter should help to ensure that the surrogate is very accurate in the 
        neighborhood of the currently sampled positions. This is done because points
        that are very close to already existing points are not kept in the memory.

        Args:
            curr_positions: The positions of the current swarm particles as array
                with shape (n_particles, dim)
            curr_f_vals: The function values of the current swarm particles as array
                with shape (n_particles,)
        """
        input_positions = np.concatenate((self.positions, curr_positions), axis=0)
        input_f_vals = np.concatenate((self.f_vals, curr_f_vals))
        self.sm.updateModel(input_positions, input_f_vals[:, None], None, None)

    def update_data(self, curr_positions, curr_f_vals, do_filtering, mean=0, std=1):
        """Updates the memory of the surrogate model.

        The default data memory is updated by this function. Either all unique points
        are accepted or unique points where the standard deviation exceeds a certain
        threshold. This memory can grow indefinitely with the number of iterations,
        which could slow the optimization immensely.

        Args:
            curr_positions: The positions of the new data points as an array with shape 
                (n, dim)
            curr_f_vals: The function values of the new data points as an array with 
                shape (n_particles,)
            do_filtering: A Boolean which determines if  filtering of the data points
                should be used
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

        The data basis for the surrogate model is fetched from the buffer memory
        and directly used for the fit.
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
        """Predicts the mean and standard deviation for given positions.

        Args:
            positions: The positions for which we want to predict the mean and std

        Returns:
            predicted_mean: Predicted mean of the given positions
            predicted_std: Predicted standard deviation of the given positions
        """

        assert positions.shape[1] == (self.dim), (
            "The dimension of the positions does \
            not match with the dimension of the model. " \
            f"Expect dimension {self.dim}, got {positions.shape[1]}")

        predicted_mean, predicted_std = self.sm.predict(positions)
        return predicted_mean, predicted_std

    def update_surrogate(self, positions, f_vals):
        """Manages the updates of the surrogate model.

        This function handles the update of the surrogate and is used on the end of the 
        optimizer. The surrogate model is fitted anew on the basis of positions and functions 
        values either with the the default memory or a buffer.

        Args:
            positions: An array of positions used to fit the surrogate model
            f_vals: An array of corresponding function values used to fit the surrogate model
        """
        if self.surrogate_options['use_buffer']:
            self.update_data_buffer(positions, f_vals)
            self.fit_model_buffer()
        else:
            mean, std = self.predict(positions)

            self.fit_model(positions, f_vals)
            self.update_data(positions, f_vals, True, mean, std)

    def get_proposition_point(self, constr):
        """Returns a proposition point.

        Given the surrogate model and the constraints of the search-space an acquisition
        function is constructed and optimized. This acquisition function values aspects
        of exploration and exploitation and is optimized with a multi-start local optimizer.
        The found point is then likely a local optimum of the acquisition function that
        could still be a good candidate for an improvement for the main optimizer.

        Args:
            constr: Constraints of the search space with shape (dim, 2)

        Returns:
            pos_proposition: Result of the acquisition function optimization
            f_val_proposition: Predicted function value of the proposed point
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
        """Implementation of the standard m proposition approach.
        
        Applies the standard m algorithm which iteratively predicts m candidates using
        the surrogate and replaces the worst m swarm elements with these points. The surrogate
        is fitted anew after each of these m iterations, so that is contains the newly proposed 
        point.

        Args:
            Swarm: A swarm instance which is used for the PSO, refer to the documentation of this
                class for further information

        Returns:
            worst_indices: The particle indices at which proposed points are stored
            other_indices: The particle indices that complement worst_indices
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

    def plotter_3d(self, constr):
        """Plots the predicted mean and variance from the surrogate.

        For the 2-dimensional case, a 3-dimensional plot is created, which shows the 
        predicted mean of the surrogate model. The predicted variance is shown in another
        3-dimensional plot.
        """
        assert self.dim == 2, f'Expected dimension to be 2! Got {self.dim} instead.'
        num = 100

        axis_x = np.linspace(constr[0, 0], constr[0, 1], num)
        axis_y = np.linspace(constr[1, 0], constr[1, 1], num)
        x, y = np.meshgrid(axis_x, axis_y)

        x = x.flatten()
        y = y.flatten()
        flat_grid = np.array([x, y]).T

        predict_mean, predict_var = self.predict(flat_grid)

        predict_mean_reshaped = np.reshape(predict_mean, (num, num))
        predict_var_reshaped = np.reshape(predict_var, (num, num))

        print(80 * "*")

        fig = go.Figure(data=[go.Surface(x=axis_x, y=axis_y, z=predict_mean_reshaped)])
        # fig.add_trace(go.Scatter3d(x=self.positions[:,0], y=self.positions[:,1], z=self.f_vals, mode='markers'))
        fig.update_layout(scene=dict(zaxis_title='predicted mean'),
                                     width=700,
                                     margin=dict(r=20, b=10, l=10, t=10))
        fig.show()

        fig = go.Figure(data=[go.Surface(x=axis_x, y=axis_y, z=predict_var_reshaped)])
        fig.update_layout(scene=dict(zaxis_title='predicted variance'),
                                     width=700,
                                     margin=dict(r=20, b=10, l=10, t=10))
        fig.show()
        print(80 * "*")
