"""Implementation of the Swarm class for the Particle Swarm Optimization (PSO)."""

from smt.sampling_methods import LHS
import numpy as np

from psoas.operations import normal_distribution, random_hypersphere_draw, uniform_distribution


class Swarm():
    """Swarm class implementation.

    Holds all information regarding the swarm used in the PSO. Most notably the current
    position and velocity of all particles and the position and function value of the personal 
    best position of each particle. Moreover it holds functions to compute the estimated 
    global optimum and a local optimum which depends on the topology.
    """

    def __init__(self, func, n_particles, dim, constr, options=None):
        """Creates and initializes a swarm class instance.

        Args:
            func: The function whose global optimum is to be determined
            n_particles: The amount of particles which is used in the swarm
            dim: The dimension of the search-space
            constr: The constraints of the search-space with shape (dim, 2)
            options: Options for the swarm in form of a dict
        """
        assert constr.shape == (dim, 2), f"Dimension of the particles ({dim}, 2) does not match the dimension of the constraints {constr.shape}!"

        if options is None:
            self.options = {"mode": 'SPSO2011', "topology":'global', "eps":0.005}
        else:
            self.options = options

        self.func = func
        self.n_particles = n_particles
        self.dim = dim
        self.constr = constr
        self._calculate_initial_values()

    def evaluate_function(self, x):
        """
        Evaluates the function specified by the class attribute self.func.

        This function asserts that the input is in shape (self.n_particles, self.dim), i.e. that 
        the given function is evaluated for each particle at the current position in the 
        self.dim-dimensional space. Therefore the output is of shape (self.n_particles,). While 
        it is generally preferable to use functions which take such input shapes and deliver such 
        a result, it is not enforced here. It is also possible to optimize on a function which 
        takes one point in the search space as the input and delivers a scalar output. This case 
        is implemented as a for loop in Python which makes it rather inefficient in comparison
        to the matrix approach.

        Args:
            x: Positions of all particles in the search space to be evaluated, shape is (self.n_particles, self.dim)
        
        Returns:
            An array with the function evaluation for each particle with shape (self.n_particles,)
        """
        assert x.shape == (self.n_particles, self.dim)
        
        try:
            res = self.func(x)
            assert res.shape == (self.n_particles,)
        except (ValueError, AssertionError):
            res = np.zeros(self.n_particles)
            for idx in range(self.n_particles):
                res[idx] = self.func(x[idx, :])

        return res

    def _calculate_initial_values(self):
        """Calculates the initial values for the position using Latin Hypercube Sampling of each
        particle. The initialization for the personal best postion and function value is given as
        a result of the initial position since it is the only position visited so far. The velocity
        for each particle is sampled uniformly between 0 and 1 (TODO: dependency on the constraints).
        """
        sampling = LHS(xlimits=self.constr) # Set up latin hypercube sampling within given constraints
        self.position = sampling(self.n_particles)

        self.velocity = (np.random.uniform(size=(self.n_particles, self.dim)) - self.position)/2

        self.pbest_position = self.position
        self.pbest = self.evaluate_function(self.position)

    def _use_topology(self):
        """
        Applies a specified topology in the computation for a local best among the particles.

        TODO: Implementation
        """
        raise NotImplementedError()

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
        if self.options['topology'] == 'global':
            gbest, gbest_position = self.compute_gbest()
            ones = np.ones(self.n_particles)
            return gbest * ones, ones[:, None] @ gbest_position[None, :]  
        else:
            raise NotImplementedError()

    def _velocity_update_SPSO2011(self):
        """
        This implementation of the velocity update is based on the Standard Particle Swarm 
        Optimization 2011 (SPSO2011) as presented in the paper ZambranoBigiarini2013 
        (doi: 10.1109/CEC.2013.6557848).
        """
        lbest, lbest_position = self.compute_lbest()
        # gbest, gbest_position = self.compute_gbest()
        
        c_1, c_2 = np.ones(2) * 0.5 + np.log(2)
       
        U_1 = uniform_distribution(self.n_particles, self.dim)
        U_2 = uniform_distribution(self.n_particles, self.dim)

        proj_pbest = self.position + c_1 * U_1 * (self.pbest_position - self.position)
        proj_lbest = self.position + c_2 * U_2 * (lbest_position - self.position) 

        center = (self.position + proj_pbest + proj_lbest) / 3

        r = np.linalg.norm(center - self.position, axis=1)

        offset = random_hypersphere_draw(r, self.dim)

        sample_points = center + offset
        
        omega = 1 / (2*np.log(2))
        self.velocity = omega * self.velocity + sample_points - self.position

    def compute_velocity(self):
        """
        TODO: docstring
        """
        if self.options['mode'] == 'SPSO2011':
            self._velocity_update_SPSO2011()
        else:
            raise NotImplementedError()
