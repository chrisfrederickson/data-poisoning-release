"""Data Oracle Defense"""

import numpy as np
from certml.utils.data import generate_class_map, get_centroids, \
    get_centroid_vec, filter_points_outside_feasible_set


class DataOracle:
    """Data Oracle Defense

    The data oracle defense determined the centroids of the data
    from a trusted data source. Then filters data outside of the
    feasible set.
    """

    def __init__(self, mode='sphere', radius=None, percentile=None):
        self.mode = mode
        self.radius = radius  # If radius is directly set, percentile is ignored.
        self.percentile = percentile
        self.radii = None
        self.centroids = None
        self.centroid_vec = None
        self.class_map = None

    def fit_trusted(self, X, y):
        """ Fit on Trusted Dataset

        Parameters
        ----------
        X : np.ndarray of shape (instances, dimensions)
            Input features
        y : np.ndarray of shape (instances,)
            Input labels
        """
        self.class_map = generate_class_map(y)
        self.centroids = get_centroids(X, y, self.class_map)
        self.centroid_vec = get_centroid_vec(self.centroids)

        if self.radius is not None:  # Radii set by absolute radius parameter
            self.radii = self.radius * np.ones((len(self.class_map),))
        elif self.percentile is not None:  # Radii set by percentile parameter
            raise NotImplementedError('Percentile radii not implemented!')
        else:
            raise ValueError('Invalid radius or radii argument!')

    def transform(self, X, y=None):
        """ Filter Input Data

        Parameters
        ----------
        X : np.ndarray of shape (instances, dimensions)
            Input features
        y : np.ndarray of shape (instances,)
            Input labels

        Returns
        -------
        x_transformed : np.ndarray of shape (instances, dimensions)
            Filtered input features
        y_transformed : np.ndarray of shape (instances,)
            Filtered input labels
        """
        if y is not None:  # Training
            if self.mode is 'sphere':
                sphere_radii = self.radii
                slab_radii = np.full(self.radii.shape, np.inf)
            elif self.mode is 'slab':
                sphere_radii = np.full(self.radii.shape, np.inf)
                slab_radii = self.radii
            else:
                raise ValueError('Invalid mode!')
            x_transformed, y_transformed = filter_points_outside_feasible_set(
                X, y, self.centroids, self.centroid_vec, sphere_radii, slab_radii, self.class_map)
            return x_transformed, y_transformed
        else:  # Testing
            return X
