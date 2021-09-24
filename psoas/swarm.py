"""Implementation of the Swarm class for the Particle Swarm Optimization (PSO)."""

import os

import numpy as np
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
import imageio

from psoas.utils import random_hypersphere_draw


class Swarm():
    """Swarm class implementation.

    Holds all information regarding the swarm used in the PSO. Most notably the current
    positions and velocities of all particles and the position and function value of the personal 
    best position of each particle. Moreover it holds functions to compute the estimated 
    global optimum and a local optimum which depends on the topology.
    """

    def __init__(self, func, n_particles, dim, constr, swarm_options, surrogate_options):
        """Creates and initializes a swarm class instance.

        Args:
            func: The function whose global optimum is to be determined
            n_particles: The amount of particles which is used in the swarm
            dim: The dimension of the search-space
            constr: The constraints of the search-space with shape (dim, 2)
            options: Options for the swarm in form of a dict
        """
        assert constr.shape == (dim, 2), f"Dimension of the particles ({dim}, 2) does not match the dimension of the constraints {constr.shape}!"

        self.swarm_options = swarm_options
        self.surrogate_options = surrogate_options  # necessary for some specific velocity updates

        self.func = func
        self.n_particles = n_particles
        self.dim = dim
        self.constr = constr
        self.constr_below = np.ones((n_particles, dim)) * constr[:, 0]
        self.constr_above = np.ones((n_particles, dim)) * constr[:, 1]
        self.velocity_reset = np.zeros((n_particles, dim))

        self._calculate_initial_values()        

        # preparation for contour plot
        if swarm_options['3d_plot'] is True:
            assert dim == 2, f'Got dim {self.dim}. Expected dim = 2.'

            self.data_plot = {}
            self.data_plot = self.get_contour(self.data_plot)
            self.gif_counter = 0
            self.gif_filenames = []

    def _calculate_initial_values(self):
        """Calculates the initial values for the position using Latin Hypercube Sampling of each
        particle. The initialization for the personal best postion and function value is given as
        a result of the initial position since it is the only position visited so far. The velocity
        for each particle is sampled uniformly between 0 and 1 (TODO: dependency on the constraints).
        """
        lhs_sampling = LHS(xlimits=self.constr) # Set up latin hypercube sampling within given constraints
        self.positions = lhs_sampling(self.n_particles)
        self.f_values = self.func(self.positions)

        self.velocities = (lhs_sampling(self.n_particles) - self.positions)/2

        self.pbest_positions = self.positions.copy()
        self.pbest = self.f_values.copy()

    def compute_gbest(self):
        """Returns the global optimum found by any of the particles."""
        idx = np.argmin(self.pbest)
        gbest_position = self.pbest_positions[idx, :]
        gbest = self.pbest[idx]

        return gbest, gbest_position

    def compute_lbest(self):
        """
        Returns the local optimum for each particle depending on the topology
        specified in the options. 
        """
        if self.swarm_options['topology'] == 'global':
            return self.topology_global()

        elif self.swarm_options['topology'] == 'ring':
            return self.topology_ring()

        elif self.swarm_options['topology'] == 'adaptive_random':
            return self.topology_adaptive_random()

        else:
            raise ValueError(f"Expected global, ring or adaptive random for the topology. Got {self.options['topology']}")

    def update(self, current_prediction=None, worst_idx=None, worst_indices=None, other_indices=None):
        """The velocity update for the swarm is calculated here and the positions of all particles
        in the swarm are updated using this new velocity. The constraints are enforced by returning 
        any particle which left the valid search space, back into it. Lastly, the personal best 
        point for each particle is updated, if the function value at the new location is better than
        the previous personal best position.
        """
        self.compute_velocity(current_prediction)
        self.enforce_constraints(check_position=False, check_velocity=True)

        if self.surrogate_options['use_surrogate']:
            # ensures that the points predicted by the surrogate do not move
            if self.surrogate_options['prediction_mode'] == 'standard':
                self.velocities[worst_idx] = 0
            elif self.surrogate_options['prediction_mode'] == 'standard_m':
                self.velocities[worst_indices] = 0

        self.positions = self.positions + self.velocities

        if self.surrogate_options['use_surrogate']:
            # reinitializes the velocity for the predicted points
            if self.surrogate_options['prediction_mode'] == 'standard':
                self.velocities[worst_idx] = np.random.normal(size=(1, self.dim))
            elif self.surrogate_options['prediction_mode'] == 'standard_m':
                m = self.surrogate_options['m']
                self.velocities[worst_indices] = np.random.normal(size=(m, self.dim))

        self.enforce_constraints(check_position=True, check_velocity=False)

        if (self.surrogate_options['use_surrogate'] and 
            self.surrogate_options['prediction_mode'] == 'standard_m'
           ):
            self.f_values[other_indices] = self.func(self.positions[other_indices])
        else:
            self.f_values = self.func(self.positions)

        # update pbest
        bool_decider = self.pbest > self.f_values

        self.pbest[bool_decider] = self.f_values[bool_decider]
        self.pbest_positions[bool_decider, :] = self.positions[bool_decider, :]

    def enforce_constraints(self, check_position, check_velocity):
        """Enforces the constraints of the valid search space.

        Any particle which left the valid search space is moved back into it. Furthermore
        the velocity which brought the particle out of the valid search space is put to 
        zero.

        TODO: replace with clip!
        """
        if check_position:
            bool_below = self.positions < self.constr[:, 0]
            bool_above = self.positions > self.constr[:, 1]

            self.positions[bool_below] = self.constr_below[bool_below]
            self.positions[bool_above] = self.constr_above[bool_above]
            self.velocities[bool_below] = self.velocity_reset[bool_below]
            self.velocities[bool_above] = self.velocity_reset[bool_above]

        if check_velocity:
            bool_below = self.velocities < self.constr[:, 0]
            bool_above = self.velocities > self.constr[:, 1]

            self.velocities[bool_below] = self.constr_below[bool_below]
            self.velocities[bool_above] = self.constr_above[bool_above]

    def _velocity_update_SPSO2011(self, current_prediction):
        """
        This implementation of the velocity update is based on the Standard Particle Swarm 
        Optimization 2011 (SPSO2011) as presented in the paper ZambranoBigiarini2013 
        (doi: 10.1109/CEC.2013.6557848).
        Args:
            current_prediction: Predicted optimum of the surroagte with shape self.dim
        """
        lbest, lbest_positions = self.compute_lbest()
        
        c_1, c_2 = np.ones(2) * 0.5 + np.log(2)
       
        U_1 = np.random.uniform(size=(self.n_particles, self.dim))
        U_2 = np.random.uniform(size=(self.n_particles, self.dim))

        proj_pbest = self.positions + c_1 * U_1 * (self.pbest_positions - self.positions)
        proj_lbest = self.positions + c_2 * U_2 * (lbest_positions - self.positions) 

        if (self.surrogate_options['use_surrogate'] and 
            self.surrogate_options['prediction_mode'] == 'center_of_gravity' and
            current_prediction is not None
           ):
            c_3 = 0.75
            U_3 = np.random.uniform(size=(self.n_particles, self.dim))
            proj_pred = self.positions + c_3 * U_3 * (current_prediction - self.positions)

            center = (self.positions + proj_pbest + proj_lbest + proj_pred) / 4

        elif (self.surrogate_options['use_surrogate'] and 
              self.surrogate_options['prediction_mode'] == 'shifting_center' and 
              current_prediction is not None
             ):
            prio = self.surrogate_options['prioritization']
            center_standard = (self.positions + proj_pbest + proj_lbest) / 3
            center = center_standard + prio * (current_prediction - center_standard)

        else:
            center = (self.positions + proj_pbest + proj_lbest) / 3           

        r = np.linalg.norm(center - self.positions, axis=1)

        offset = random_hypersphere_draw(r, self.dim)

        sample_points = center + offset
        
        omega = 1 / (2*np.log(2))
        self.velocities = omega * self.velocities + sample_points - self.positions

    def _velocity_update_MSPSO2011(self):
        """
        This implementation of the velocity update is based on the paper Hariya2016, 
        which in itself based on the SPSO2011.
        """
        lbest, lbest_positions = self.compute_lbest()

        c_1, c_2 = np.ones(2) * 0.5 + np.log(2)
        omega = 1 / (2*np.log(2))

        comp_identity = 2*np.ones((self.n_particles, self.dim))

        U_1 = np.random.uniform(size=(self.n_particles, self.dim))

        proj_pbest = self.positions + c_1 * 2 * U_1 * (self.pbest_positions - self.positions)
        proj_lbest = self.positions + c_2 * (comp_identity - 2 * U_1) * (lbest_positions - self.positions)

        center = (self.positions + proj_pbest + proj_lbest) / 3

        r = np.linalg.norm(center - self.positions, axis=1)

        offset = random_hypersphere_draw(r, self.dim)

        sample_points = center + offset
        
        self.velocities = omega * self.velocities + sample_points - self.positions

    def compute_velocity(self, current_prediction):
        """
        TODO: docstring
        """
        if self.swarm_options['mode'] == 'SPSO2011':
            self._velocity_update_SPSO2011(current_prediction)

        elif self.swarm_options['mode'] == 'MSPSO2011':
            self._velocity_update_MSPSO2011()

        else:
            
            raise NotImplementedError()

    def topology_global(self):
        """
        Implements a global exchange of the personal bests between the particles.
        """
        gbest, gbest_position = self.compute_gbest()
        ones = np.ones(self.n_particles)
        return gbest * ones, ones[:, None] @ gbest_position[None, :]

    def topology_ring(self):
        """
        Implements a exchange of personal bests according to a ringtopology.
        """
        neighbors = np.zeros([self.n_particles, 3])
        neighbors[0, 0] = self.pbest[-1]
        neighbors[1:, 0] = self.pbest[0:-1]
        neighbors[:, 1]  = self.pbest
        neighbors[-1, 2] = self.pbest[0]
        neighbors[:-1, 2] = self.pbest[1:]

        best_indices = np.argmin(neighbors, axis=1)
        lbest = np.choose(best_indices, neighbors.T)

        pos_indices = np.linspace(0, self.n_particles-1, self.n_particles, dtype=np.int32) + best_indices - 1

        #ensure index wrapping
        if pos_indices[-1] == self.n_particles:
            pos_indices[-1] = 0
        lbest_positions = self.pbest_positions[pos_indices]

        return lbest, lbest_positions

    def topology_adaptive_random(self):
        """
        Implements a exchange of personal bests in a random fashion.
        """
        n_neighbors = 3

        # the assignments are changed if there is no change in the global best
        update_neighbors = self.no_change_in_gbest

        if (not hasattr(self, 'neighbors')) or update_neighbors:
            self.neighbors = np.random.rand(self.n_particles, self.n_particles).argpartition(n_neighbors, axis=1)[:,:n_neighbors]
            self.neighbors = np.concatenate((np.arange(0, self.n_particles)[:, None], self.neighbors), axis=1)

        informed_particles = np.zeros((self.n_particles, self.n_particles))
        informed_particles[:] = np.nan
        for i in range(self.n_particles):
            informed_indices = self.neighbors[i]
            for idx in informed_indices:
                informed_particles[idx, i] = self.pbest[i]

        best_indices = np.argmin(informed_particles, axis=1)
        return self.pbest[best_indices], self.pbest_positions[best_indices]

    def get_contour(self, data_plot):
        """
        Generates data for plotting the current function.

        Args:
            data_plot: Dict to store the data necessary for plotting
        
        Returns:
            Returns the input dict with three keys: x, y and z
        """
        delta = 0.1
        B = np.arange(-100, 100, delta)
        data_plot['x'] = B
        data_plot['y'] = B

        xx, yy = np.meshgrid(B,B, sparse=True)
        data_plot['z'] = np.zeros((xx.shape[1], yy.shape[0]))

        for i in range(xx.shape[1]):
            for j in range(yy.shape[0]):
                data_plot['z'][i,j] = self.func.function(np.array([xx[0][i], yy[j][0]]))
        return data_plot

    def plotter(self):
        """
        Plotting the current function.
        """
        plt.plot(self.positions[:,0], self.positions[:,1], 'o')
        plt.contourf(self.data_plot['x'], self.data_plot['y'], self.data_plot['z'])
        plt.quiver(self.positions[:,0], self.positions[:,1], self.velocities[:,0], self.velocities[:,1], units='xy', scale_units='xy', scale=1)

        plt.xlim((-100, 100))
        plt.ylim((-100, 100))

        plt.xlabel("x")
        plt.ylabel("y")

        if self.swarm_options['create_gif']:
            filename = f'{self.gif_counter}.png'
            self.gif_filenames.append(filename)
    
            # save frame
            plt.savefig(filename)
            plt.close()
            self.gif_counter += 1
        plt.show()
    
    def create_gif(self):
        """
        Create a gif of the particles moving through the cost landscape.
        """
        with imageio.get_writer('PSO.gif', mode='I') as writer:
            for filename in self.gif_filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        print('Gif has been written.')
        
        # Remove files
        for filename in set(self.gif_filenames):
            os.remove(filename)
