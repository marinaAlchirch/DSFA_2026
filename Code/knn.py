import numpy as np
import pandas as pd

from utils import get_dimensions


class KNearestNeighborsRegressor:

    def __init__(self, window, k=1, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.window = window

    def partial_fit(self, X, y, sample_weight=None): # From knn_regressor.py (skmultiflow)
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: numpy.ndarray of shape (n_samples)
            An array-like containing the target values for all
            samples in X.

        sample_weight: Not used.

        Returns
        -------
        KNNRegressor
            self

        Notes
        -----
        For the K-Nearest Neighbors regressor, fitting the model is the
        equivalent of inserting the newer samples in the observed window,
        and if the size_limit is reached, removing older results.

        """
        r, c = get_dimensions(X)

        for i in range(r):
            self.window.add_sample(X=X[i], y=y[i])
        return self


    def predict(self, X):
        predictions = []

        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert to numpy array

        for i in range(len(X)):
            instance_features = X[i].reshape(1, -1)
            predictions.append(self._predict(X_test=instance_features))

        return predictions

    def _predict(self, X_test):
        distances = []
        for i in range(len(self.window)):
            instance_features = self.window.features_buffer[i]
            if not np.array_equal(instance_features, X_test):
                distances.append(self.distance(instance_features.reshape(1,-1), X_test, name=self.distance_metric))
        if len(distances) < self.k:
            return 0.0 # return default prediction if k is larger than window


        neighbors_indices = self.find_indices_of_k_smallest_elements(input_list=distances, k=self.k)

        k_nearest_neighbors = [self.window.targets_buffer[j] for j in neighbors_indices]
        #
        # # Regression prediction is the mean of the targets of the k neighbors
        prediction = np.mean(k_nearest_neighbors)
        return prediction


    def distance(self, instance_features, X_test, name='euclidean'):
        if name == 'euclidean':
            print("euclidean")
            return np.sum(np.sqrt((instance_features-X_test)**2))
        if name == 'manhattan':
            return np.sum(np.abs(X_test - instance_features))


    def find_indices_of_k_smallest_elements(self, input_list, k):
        indices = []
        for i in range(k):
            index = np.argmin(input_list)
            indices.append(index)
            input_list[index] = np.inf
        return indices
