"""Implementation of the Swarm class for the Particle Swarm Optimization (PSO)."""

import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import numpy as np
import os
import imageio

from psoas.operations import normal_distribution, random_hypersphere_draw, uniform_distribution


class Swarm():
    """Swarm class implementation.

    Holds all information regarding the swarm used in the PSO. Most notably the current
    position and velocity of all particles and the position and function value of the personal 
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
        self.surrogate_options = surrogate_options

        self.func = func
        self.n_particles = n_particles
        self.dim = dim
        self.constr = constr
        
        self._calculate_initial_values()        

        # preparation for contour plot
        if swarm_options['3d_plot'] is True:
            assert dim == 2, f'Got dim {self.dim}. Expect dim 2.'

            self.data_plot = {}
            self.data_plot = self.get_contour(self.data_plot)
            self.gif_counter = 0
            self.gif_filenames = []

    def evaluate_function(self, x):
        """
        Evaluates the function specified by the class attribute self.func.

        This function asserts that the input is in shape (self.n_particles, self.dim), i.e. that 
        the given function is evaluated for each particle at the current position in the 
        self.dim-dimensional space. Therefore the output is of shape (self.n_particles,).

        Args:
            x: Positions of all particles in the search space to be evaluated, shape is (self.n_particles, self.dim)
        
        Returns:
            An array with the function evaluation for each particle with shape (self.n_particles,)
        """
        assert x.shape == (self.n_particles, self.dim)

        res = self.func(x)
        assert res.shape == (self.n_particles,)

        return res

    def _calculate_initial_values(self):
        """Calculates the initial values for the position using Latin Hypercube Sampling of each
        particle. The initialization for the personal best postion and function value is given as
        a result of the initial position since it is the only position visited so far. The velocity
        for each particle is sampled uniformly between 0 and 1 (TODO: dependency on the constraints).
        """
        lhs_sampling = LHS(xlimits=self.constr) # Set up latin hypercube sampling within given constraints
        self.position = lhs_sampling(self.n_particles)
        self.f_values = self.evaluate_function(self.position)

        self.velocity = (lhs_sampling(self.n_particles) - self.position)/2

        self.pbest_position = self.position
        self.pbest = self.f_values

    def compute_gbest(self):
        """Returns the global optimum found by any of the particles."""
        idx = np.argmin(self.pbest)
        gbest_position = self.pbest_position[idx, :]
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

    def _velocity_update_SPSO2011(self, current_prediction):
        """
        This implementation of the velocity update is based on the Standard Particle Swarm 
        Optimization 2011 (SPSO2011) as presented in the paper ZambranoBigiarini2013 
        (doi: 10.1109/CEC.2013.6557848).
        """
        lbest, lbest_position = self.compute_lbest()
        
        c_1, c_2 = np.ones(2) * 0.5 + np.log(2)
       
        U_1 = uniform_distribution(self.n_particles, self.dim)
        U_2 = uniform_distribution(self.n_particles, self.dim)

        proj_pbest = self.position + c_1 * U_1 * (self.pbest_position - self.position)
        proj_lbest = self.position + c_2 * U_2 * (lbest_position - self.position) 

        if (self.surrogate_options['use_surrogate'] and 
            self.surrogate_options['prediction_mode'] == 'center_of_gravity' and
            current_prediction is not None
           ):
            c_3 = 0.75
            U_3 = uniform_distribution(self.n_particles, self.dim)
            proj_pred = self.position + c_3 * U_3 * (current_prediction - self.position)

            center = (self.position + proj_pbest + proj_lbest + proj_pred) / 4

        elif (self.surrogate_options['use_surrogate'] and 
              self.surrogate_options['prediction_mode'] == 'shifting_center' and 
              current_prediction is not None
             ):
            prio = self.surrogate_options['prioritization']
            center_standard = (self.position + proj_pbest + proj_lbest) / 3
            center = center_standard + prio * (current_prediction - center_standard)

        else:
            center = (self.position + proj_pbest + proj_lbest) / 3           

        r = np.linalg.norm(center - self.position, axis=1)

        offset = random_hypersphere_draw(r, self.dim)

        sample_points = center + offset
        
        omega = 1 / (2*np.log(2))
        self.velocity = omega * self.velocity + sample_points - self.position

    def _velocity_update_MSPSO2011(self):
        """
        This implementation of the velocity update is based on the paper Hariya2016, based on the SPSO2011.
        """
        lbest, lbest_position = self.compute_lbest()

        c_1, c_2 = np.ones(2) * 0.5 + np.log(2)
        omega = 1 / (2*np.log(2))

        comp_identity = 2*np.ones((self.n_particles, self.dim))

        U_1 = uniform_distribution(self.n_particles, self.dim)

        proj_pbest = self.position + c_1 * 2 * U_1 * (self.pbest_position - self.position)
        proj_lbest = self.position + c_2 * (comp_identity - 2 * U_1) * (lbest_position - self.position)

        center = (self.position + proj_pbest + proj_lbest) / 3

        r = np.linalg.norm(center - self.position, axis=1)

        offset = random_hypersphere_draw(r, self.dim)

        sample_points = center + offset
        
        self.velocity = omega * self.velocity + sample_points - self.position

    def compute_velocity(self):
        """
        TODO: docstring
        """
        if self.swarm_options['mode'] == 'SPSO2011':
            self._velocity_update_SPSO2011()

        elif self.swarm_options['mode'] == 'MSPSO2011':
            self._velocity_update_MSPSO2011()

        else:
            
            raise NotImplementedError()

    def topology_global(self):
        """
        TODO: docstring
        """
        gbest, gbest_position = self.compute_gbest()
        ones = np.ones(self.n_particles)
        return gbest * ones, ones[:, None] @ gbest_position[None, :]

    def topology_ring(self):
        """
        TODO: docstring
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
        lbest_position = self.pbest_position[pos_indices]

        return lbest, lbest_position

    def topology_adaptive_random(self):
        """
        TODO: docstring
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
        return self.pbest[best_indices], self.pbest_position[best_indices]

    def get_contour(self, data_plot):
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
        TODO: docstring
        """
        plt.plot(self.position[:,0], self.position[:,1], 'o')
        plt.contourf(self.data_plot['x'], self.data_plot['y'], self.data_plot['z'])
        plt.quiver(self.position[:,0], self.position[:,1], self.velocity[:,0], self.velocity[:,1], units='xy', scale_units='xy', scale=1)

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
        TODO: docstring
        """
        with imageio.get_writer('PSO.gif', mode='I') as writer:
            for filename in self.gif_filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        print('Gif has been written.')
        
        # Remove files
        for filename in set(self.gif_filenames):
            os.remove(filename)
