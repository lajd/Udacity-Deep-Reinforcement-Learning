import numpy as np


class SpaceDiscretizer:
    def __init__(self, dimension_lower_bounds, dimension_upper_bounds, dimension_intervals=(10, 10), ):
        self.dimension_lower_bounds = dimension_lower_bounds
        self.dimension_upper_bounds = dimension_upper_bounds
        self.dimension_intervals = dimension_intervals
        self.resolved_space = self.create_uniform_grid()

    def create_uniform_grid(self):
        """Define a uniformly-spaced grid that can be used to discretize a space.

        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of the continuous space.
        high : array_like
            Upper bounds for each dimension of the continuous space.
        bins : tuple
            Number of bins along each corresponding dimension.

        Returns
        -------
        grid : list of array_like
            A list of arrays containing split points for each dimension.
        """
        grid = []
        for dim_idx, num_dim_intervals in enumerate(self.dimension_intervals):
            grid.append(
                np.linspace(
                    self.dimension_lower_bounds[dim_idx],
                    self.dimension_upper_bounds[dim_idx],
                    num_dim_intervals + 1
                )[1:-1]
            )
        return grid

    def discretize(self, sample):
        """Discretize a sample as per given grid.

        Parameters
        ----------
        sample : array_like
            A single sample from the (original) continuous space.
        grid : list of array_like
            A list of arrays containing split points for each dimension.

        Returns
        -------
        discretized_sample : array_like
            A sequence of integers with the same number of dimensions as sample.
        """
        discretized_sample = []
        for dim_sample_value, dim_grid in zip(sample, self.resolved_space):
            # Convert to discrete state
            discretized_sample.append(int(np.digitize(dim_sample_value, dim_grid)))
        return discretized_sample
