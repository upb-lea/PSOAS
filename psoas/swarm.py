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
    positions, corresponding function values and velocities of all particles and the 
    positions and function values of the personal best position for each particle. It 
    contains functions to perform an iterative step and update all of these values.
    Moreover it contains functions to compute the estimated global optimum in a given 
    iteration and a local optimum for each particle (which depends on the topology).

    Attributes:
        swarm_options: Dict containing options that belong to the swarm
        surrogate_options: Dict containing options that belong to the surrogate
            (this is necessary for some specific velocity updates)
        func: The function to be optimized
        n_particles: The number of particles in the swarm
        dim: The dimension of the search-space
        constr: The constraints of the search-space with shape (dim, 2)
        positions: The positions of the particles in dim-dimensional space with
            shape (n_particles, dim)
        f_values: The function values of the particles with shape (n_particles, 1)
        velocities: The velocities of the particles with shape (n_particles, dim)
        pbest_positions: The best position so far per particle with shape (n_particles, dim)
        pbest: The function value of the best position so far with shape (n_particles, 1)
    """

    def __init__(self, func, n_particles, dim, constr, swarm_options, surrogate_options):
        """Creates and initializes a swarm class instance.

        The function calculates initial values for the positions and velocities of the
        particles using latin hypercube sampling and uses its inputs to initialize most 
        of the swarms other attributes.

        Args:
            func: The function to be optimized
            n_particles: The number of particles in the swarm
            dim: The dimension of the search-space
            constr: The constraints of the search-space with shape (dim, 2)
            swarm_options: Dict containing options that belong to the swarm
            surrogate_options: Dict containing options that belong to the surrogate
                (this is necessary for some specific velocity updates)
        """
        assert constr.shape == (dim, 2), \
            f"Dimension of the particles ({dim}, 2) does not match the dimension of the constraints {constr.shape}!"

        self.swarm_options = swarm_options
        self.surrogate_options = surrogate_options

        self.func = func
        self.n_particles = n_particles
        self.dim = dim
        self.constr = constr
        self._velocity_reset = np.zeros((n_particles, dim))

        self._calculate_initial_values()        

        # preparation for contour plot
        if self.swarm_options['3d_plot'] is True:
            assert self.dim == 2, f'Got dim {self.dim}. Expected dim = 2.'

            self._data_plot = self._get_contour()
            self._gif_counter = 0
            self._gif_filenames = []

    def _calculate_initial_values(self):
        """Calculates initial values for positions, function values and velocities.
        
        Calculates the initial values for the positions and the velocities using Latin 
        Hypercube Sampling for each particle. The function values are then calculated based
        on the initial position. The initialization for the personal best postion and 
        function value is given as a result of the initial position since it is the only 
        position visited so far.
        """
        lhs_sampling = LHS(xlimits=self.constr) # Set up latin hypercube sampling within given constraints
        self.positions = lhs_sampling(self.n_particles)
        self.f_values = self.func(self.positions)

        self.velocities = (lhs_sampling(self.n_particles) - self.positions)/2

        self.pbest_positions = self.positions.copy()
        self.pbest = self.f_values.copy()

    def compute_gbest(self):
        """Returns the function value and the position of the best point found so far."""
        idx = np.argmin(self.pbest)
        gbest_position = self.pbest_positions[idx, :]
        gbest = self.pbest[idx]

        return gbest, gbest_position

    def compute_lbest(self):
        """Returns the local optimum for each particle depending on the topology
        specified in the options. 
        """
        if self.swarm_options['topology'] == 'global':
            return self._topology_global()

        elif self.swarm_options['topology'] == 'ring':
            return self._topology_ring()

        elif self.swarm_options['topology'] == 'adaptive_random':
            return self._topology_adaptive_random()

        else:
            raise ValueError(f"Expected global, ring or adaptive random for the topology. Got {self.options['topology']}")

    def update(self, current_prediction=None, worst_idx=None, worst_indices=None, other_indices=None):
        """Performs one iterative update step for the Swarm.
        
        The velocity update for the swarm is calculated here and the positions of all 
        particles in the swarm are updated using this new velocity. The constraints are
        enforced by returning any particle which left the valid search space, back into 
        it. Lastly, the personal best point for each particle is updated, if the function 
        value at the new location is better than the previous personal best position. 
        
        For certain surrogate prediction modes, the proposed points are saved within the
        swarm but were not used yet (used in the fit of the surrogate or compared to the 
        personal bests of the particles). These points are held in place by setting the 
        respective velocities to zero and reinitializing the velocities after the update 
        of the positions. In the case of the standard_m prediction mode, the function values
        are already updated and their calculation can therefore be skipped in this update step.

        Args:
            current_prediction: The last point that was proposed by the surrogate (Only 
                necessary for surrogate prediction modes center_of_gravity and 
                shifting_center)
            worst_idx: The particle index at which a proposed point is stored (Only for
                standard surrogate prediction mode)
            worst_indices: The particle indices at which proposed points are stored
                (Only for standard_m surrogate prediction mode)
            other_indices: The particle indices that complement worst_indices (Only for
                standard_m surrogate prediction mode)
        """
        self.compute_velocity(current_prediction)
        self.enforce_constraints(check_position=False, check_velocity=True)

        if self.surrogate_options['use_surrogate']:
            # ensures that the points proposed by the surrogate do not move
            if self.surrogate_options['prediction_mode'] == 'standard':
                self.velocities[worst_idx] = 0
            elif self.surrogate_options['prediction_mode'] == 'standard_m':
                self.velocities[worst_indices] = 0

        self.positions = self.positions + self.velocities

        if self.surrogate_options['use_surrogate']:
            # reinitializes the velocity for the proposed points
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

        If the positional constraints are enforced, any particle which left the valid search
        space is moved back into it. Furthermore the velocity which brought the particle out 
        of the valid search space will then be put to zero. 
        If the velocity constraints are enforced, the velocities of all particles will be 
        clipped to the constraints. 

        Args:
            check_position: Flag which determines if the positional constraints should be 
                enforced
            check_velocity: Flag which determines if the velocity constraints should be 
                enforced
        """
        if check_position:
            bool_below = self.positions < self.constr[:, 0]
            bool_above = self.positions > self.constr[:, 1]

            self.positions = np.clip(self.positions, self.constr[:, 0], self.constr[:, 1])

            self.velocities[bool_below] = self._velocity_reset[bool_below]
            self.velocities[bool_above] = self._velocity_reset[bool_above]

        if check_velocity:
            self.velocities = np.clip(self.velocities, self.constr[:, 0], self.constr[:, 1])

    def _velocity_update_SPSO2011(self, current_prediction):
        """Calculates a new velocity for each of the particles using the SPSPO2011 update rule.

        This implementation of the velocity update is based on the Standard Particle Swarm 
        Optimization 2011 (SPSO2011) as presented in the paper ZambranoBigiarini2013 
        (doi: 10.1109/CEC.2013.6557848). The update is slightly modified for shifting_center
        and center_of_gravity surrogate prediction approaches. The documentation on gitlab
        should be consulted for further details on the modified update rules.

        Args:
            current_prediction: The last point that was proposed by the surrogate (Only 
                for surrogate prediction modes center_of_gravity and shifting_center)
        """
        lbest, lbest_positions = self.compute_lbest()
        
        c_1, c_2 = np.ones(2) * 0.5 + np.log(2)
        U_1 = np.random.uniform(size=(self.n_particles, self.dim))
        U_2 = np.random.uniform(size=(self.n_particles, self.dim))
        proj_pbest = self.positions + c_1 * U_1 * (self.pbest_positions - self.positions)
        proj_lbest = self.positions + c_2 * U_2 * (lbest_positions - self.positions) 

        # modified center calculation for the center_of_gravity approach
        if (self.surrogate_options['use_surrogate'] and 
            self.surrogate_options['prediction_mode'] == 'center_of_gravity' and
            current_prediction is not None
           ):
            c_3 = 0.75
            U_3 = np.random.uniform(size=(self.n_particles, self.dim))
            proj_pred = self.positions + c_3 * U_3 * (current_prediction - self.positions)

            center = (self.positions + proj_pbest + proj_lbest + proj_pred) / 4

        # modified center calculation for the shifting_center approach
        elif (self.surrogate_options['use_surrogate'] and 
              self.surrogate_options['prediction_mode'] == 'shifting_center' and 
              current_prediction is not None
             ):
            prio = self.surrogate_options['prioritization']
            center_standard = (self.positions + proj_pbest + proj_lbest) / 3
            center = center_standard + prio * (current_prediction - center_standard)

        # standard SPSO2011 center calculation
        else:
            center = (self.positions + proj_pbest + proj_lbest) / 3           

        r = np.linalg.norm(center - self.positions, axis=1)
        offset = random_hypersphere_draw(r, self.dim)
        sample_points = center + offset
        omega = 1 / (2*np.log(2))
        self.velocities = omega * self.velocities + sample_points - self.positions

    def _velocity_update_MSPSO2011(self):
        """Calculates a new velocity for each of the particles using the MSPSPO2011 update rule.

        This implementation of the velocity update is based on the paper Hariya2016, 
        which in itself based on the SPSO2011 (doi: 10.1109/CEC.2016.7744012).
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
        """Wrapper for the different velocity updates.

        Depending on the choice in the options one of the velocity updates is performed.
        Generally, it is adviseable to use the SPSO2011 since it usually performs better 
        than the MSPSO2011.

        Args:
            current_prediction: The last point that was proposed by the surrogate (Only 
                for surrogate prediction modes center_of_gravity and shifting_center)
        """
        if self.swarm_options['mode'] == 'SPSO2011':
            self._velocity_update_SPSO2011(current_prediction)

        elif self.swarm_options['mode'] == 'MSPSO2011':
            self._velocity_update_MSPSO2011()

        else:
            raise ValueError(f"Expected SPSO2011 or MSPSO2011 for the swarm mode. Got {self.options['mode']}")

    def _topology_global(self):
        """Implements the global exchange of the personal bests between the particles.

        With this topology the global optimum of the whole swarm is directly known by
        any particle in the swarm.
        """
        gbest, gbest_position = self.compute_gbest()
        ones = np.ones(self.n_particles)
        return gbest * ones, ones[:, None] @ gbest_position[None, :]

    def _topology_ring(self):
        """Implements the exchange of personal bests according to a ring topology.

        With this topology the personal optima found by any of the particles are shared 
        with two other particles of the swarm whose indices are one above and one below
        the particle.
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

    def _topology_adaptive_random(self, n_neighbors=3):
        """Implements the exchange of personal bests with changing random partners.

        A set of neighbors is assigned to each particle and the particle informs its
        neighbors with its personal optimum. The assignment is made once in the 
        beginning and is then changed whenever the global optimum shows no improvement 
        from one iteration to the next. Take a look at the paper ZambranoBigiarini2013 
        (doi: 10.1109/CEC.2013.6557848) for another description.
        """
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

    # plotting ##########################################################################

    def _get_contour(self):
        """Generates contours for the current function which are used to create a plot 
        and possibly a gif of the swarm moving over the function (Only usable for 
        dimension=2).

        Returns:
            Returns a dict containing the function values at key 'z' and the x and y
                values at keys 'x' and 'y' respectively
        """
        data_plot = {}
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

    def _plotter(self):
        """Creates a contour plot of the current function while showing the position and
        velocity of all particles.
        """
        plt.plot(self.positions[:, 0], self.positions[:, 1], 'o')
        plt.contourf(self._data_plot['x'], self._data_plot['y'], self._data_plot['z'])
        plt.quiver(self.positions[:, 0], self.positions[:, 1], self.velocities[:, 0], self.velocities[:, 1], 
                   units='xy', scale_units='xy', scale=1)

        plt.xlim((-100, 100))
        plt.ylim((-100, 100))

        plt.xlabel("x")
        plt.ylabel("y")

        if self.swarm_options['create_gif']:
            filename = f'{self._gif_counter}.png'
            self._gif_filenames.append(filename)
    
            # save frame
            plt.savefig(filename)
            plt.close()
            self._gif_counter += 1
        plt.show()
    
    def _create_gif(self):
        """Create a gif of the particles moving through the contour plot."""

        with imageio.get_writer('PSO.gif', mode='I') as writer:
            for filename in self._gif_filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        print('Gif has been written.')
        
        # Remove files
        for filename in set(self._gif_filenames):
            os.remove(filename)
