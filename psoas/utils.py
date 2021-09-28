"""Collection of basic utility functions and classes."""

import os

import numpy as np
import matplotlib.pyplot as plt
import imageio
import shutil


def random_hypersphere_draw(r, dim):
    """Uniform sampling from dim-dimensional hyperspheres.

    The goal is to sample a dim-dimensional array for each particle. One sample is
    drawn uniformly from a hypersphere where the radius corresponds to one element 
    of r. A detailed explanation can be found in the paper Voelker2017
    (doi: 10.13140/RG.2.2.15829.01767/1).

    Args:
        r: Array of shape (n_particles,) containing the different radii, one for 
            each particle
        dim: The dimension of the search-space

    Returns:
        u: Samples with shape (n_particles, dim) drawn uniformly from a dim-dimensional 
            hypersphere
    """
    u = np.random.normal(size=(r.shape[0], dim+2))
    u_norm = np.linalg.norm(u, axis=1)
    u /= u_norm[:,None]
    u *= r[:,None]
    return u[:,:dim]


def calc_max_iter(max_f_eval, n_particles):
    """Calculate the maximum iterations as a function of the maximum function evaluations.

    For some applications, a fixed budget of functions evaluations must be considered. 
    This function calculates the maximum possible iterations given the budget and the 
    number of particles. It is ensured that the number of actual function evaluation 
    does not exceed the given budget, even if that means that the budget is not used
    completely.
    
    Args:
        max_f_eval: Maximum number of function evaluations
        n_particles: The number of particles used in the swarm

    Returns:
        max_iter: Maximum number of interations
    """
    max_iter = np.floor((max_f_eval - n_particles)/n_particles)
    return max_iter.astype('int')


class counting_function():
    """The given function is extendend such that the number of function calls is counted.

    Attributes:
        eval_count: Number of function evaluations
        function: The actual function which is wrapped by this class
    """
    def __init__(self, function):
        self.eval_count = 0
        self.function = function

    def __call__(self, x):
        self.eval_count += 1
        return self.function(x)


class counting_function_cec2013_single(counting_function):
    """Specific extension to the counting_function. It takes array-like inputs of shape
    (N, dimension) and calculates the function for each of the N rows. Naturally the
    function counter is incremented by N. This is helpful for functions that can not
    deal with array-like inputs such as the cec2013 test functions on which this 
    optimizer was benchmarked.

    Attributes:
        eval_count: Number of function evaluations
        function: The actual function which is wrapped by this class
    """
    def __call__(self, x):
        length = x.shape[0]
        self.eval_count += length

        res = np.zeros(length)
        for idx in range(length):
            res[idx] = self.function(x[idx, :])
        return res


class TimeDataBuffer:
    """Implementation of the time data buffer.

    This class implements a ringbuffer that works according to the first in, first out
    (FIFO) principle. The positions and associated function values are stored in one
    slot, while the number of slots can be set.

    Attributes:
        position_buffer: Array of shape (n_slots * n_particles, dim) containing the
            positions of the last n_slots-iterations
        f_val_buffer: Array of shape (n_slots * n_particles, 1) containing the function
            values of the last n_slots-iterations

    """
    def __init__(self, dim, n_particles, n_slots=4):
        """Creates and initializes a time data buffer class instance.
        
        Args:
            dim: The dimension of the search-space
            n_particles: The number of particles used in the swarm
            n_slots: The number of slots used for storage, one slot contains the data of
                one iteration
        """

        self.position_buffer = np.zeros((n_slots * n_particles, dim), dtype=np.float32)
        self.f_val_buffer = np.zeros((n_slots * n_particles, 1), dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._max_size = n_slots
        self._n_particles = n_particles
    
    def store(self, positions, f_vals):
        """Store the positions and corresponding function values.

        The data of one iteration is given to the buffer and the oldest data is replaced
        with this new data.

        Args:
            positions: The positions of the particles in with shape (n_particles, dim)
            f_vals: The function values of the particles with shape (n_particles, 1)
        """
        start = self._ptr * self._n_particles
        end = (self._ptr+1) * self._n_particles

        self.position_buffer[start:end] = positions.copy()
        self.f_val_buffer[start:end] = f_vals.copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def fetch(self):
        """Fetches the positions and corresponding function values.

        If the data buffer is fully filled one could also directly access the position_buffer
        and f_val_buffer, but as long as the buffer is not fully filled this function will 
        only return valid values while direct access will return the zeros with which empty 
        slots are initialized.

        Returns:
            postions: Array of shape (n_slots * n_particles, dim) containing the
                positions of the buffer
            f_vals: Array of shape (n_slots * n_particles, 1) containing the
                function values of the buffer
        """
        positions = self.position_buffer[:self._size*self._n_particles]
        f_vals = self.f_val_buffer[:self._size*self._n_particles]
        return positions, f_vals


class ValueDataBuffer:
    def __init__(self, dim, buffer_size):
        self.position_buffer = np.zeros((buffer_size, dim), dtype=np.float32)
        self.f_val_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = buffer_size
    
    def store(self, positions, f_vals):
        length = positions.shape[0]        

        for i in range(length):
            
            # check if the position is already stored in the buffer
            already_in_buffer = (positions[i] == self.position_buffer).all(axis=-1).any()
            if already_in_buffer:
                continue
            
            # if the buffer is not full yet, the new point is added 
            # regardless of its value
            if self.size < self.max_size:
                self.position_buffer[self.ptr] = positions[i]
                self.f_val_buffer[self.ptr] = f_vals[i]
                self.ptr += 1
                self.size += 1
            
            # if the buffer is full, check if the new point is better
            # than the current worst point 
            else:
                idx_max = np.argmax(self.f_val_buffer)
                f_val_max = self.f_val_buffer[idx_max]
                if f_vals[i] < f_val_max:
                    self.position_buffer[idx_max] = positions[i]
                    self.f_val_buffer[idx_max] = f_vals[i]
    
    def fetch(self):
        positions = self.position_buffer[:self.size]
        f_vals = self.f_val_buffer[:self.size]
        return positions, f_vals


class SwarmPlotter:
    def __init__(self, func):
        """Generates contours for the current function which are used to create a plot 
        and possibly a gif of the swarm moving over the function (Only usable for 
        dimension=2).
        A dict containing the function values at key 'z' and the x and y values at keys 
        'x' and 'y' respectively is created in the process.

        WARNING: Can be computationally quite demanding depending on the function. If
        array inputs are possible for your function, it might be helpful to replace the
        for loop with an array operation.
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
                data_plot['z'][i,j] = func.function(np.array([xx[0][i], yy[j][0]]))

        self.data_plot = data_plot
        
        # create a temporary dir to store the images
        _cwd = os.getcwd()
        self._path = '/tmp_gif'
        self._tmp_path = _cwd + self._path

        if os.path.isdir(self._tmp_path):
            shutil.rmtree(self._tmp_path)

        os.mkdir(self._tmp_path)

        self._gif_counter = 0
        self._gif_filenames = []

    def plot(self, positions, velocities, create_gif=False):
        """Creates a contour plot of the current function while showing the position and
        velocity of all particles.
        """
        plt.plot(positions[:, 0], positions[:, 1], 'o')
        plt.contourf(self.data_plot['x'], self.data_plot['y'], self.data_plot['z'])
        plt.quiver(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1], 
                   units='xy', scale_units='xy', scale=1)

        plt.xlim((-100, 100))
        plt.ylim((-100, 100))

        plt.xlabel("x")
        plt.ylabel("y")

        if create_gif:
            filename = f'{self._gif_counter}.png'
            self._gif_filenames.append(filename)
    
            # save frame
            plt.savefig(f'{self._tmp_path}/{filename}')
            plt.close()
            self._gif_counter += 1
        plt.show()
    
    def create_gif(self):
        """Create a gif of the particles moving through the contour plot."""

        with imageio.get_writer('PSO.gif', mode='I') as writer:
            for filename in self._gif_filenames:
                image = imageio.imread(f'{self._tmp_path}/{filename}')
                writer.append_data(image)

        print('Gif has been written.')
        
        # remove files
        for filename in set(self._gif_filenames):
            os.remove(f'{self._tmp_path}/{filename}')
        os.rmdir(self._tmp_path)

