"Marina code"
from src.utils import get_dimensions

"""Here we devide keep two sliding windows per node, one that has the examples (statistics)
that belong to majority classes (most frequent classes through the visited examples
and the other the opposite (examples in minority classes


Next step : While doing this in a Hoeffding based tree, each time that a new instance gets sorted we check 
if the statistics table (the two sliding windows) is imbalanced 
(we need to find a paper (algorithm) that acts as a checker of imbalance

My idea for an imbalance checker : have a threshold of allowing e.g. 40% of minority classes, 
most probably the bound will be 40%, that is if there are 40% examples of minority class then we decide 
that the dataset is imbalanced."""




from src.utils.data_structures import SlidingWindow

import numpy as np



class BalancedWindow(SlidingWindow):
    """Marina note : keep a window that keeps the statistics of examples"""

    """ Keep a fixed-size sliding window of the most recent data samples.

        Parameters
        ----------

        window_size: int, optional (default=1000)
            The window's size.

        Raises
        ------
        ValueError
            If at any moment, a sample with a different number of attributes than
             those already observed is passed.

        Notes
        -----
        It updates its stored samples by the FIFO method, which means
        that when size limit is reached, old samples are dumped to give
        place to new samples.

        The internal buffer does not keep order of the stored samples,
        when the size limit is reached, the older samples are overwritten
        with new ones (circular buffer).

        """



    def __init__(self, window_size=1000):
        super().__init__()

        self.window_size = window_size
        self._n_features = -1
        self._n_targets = -1
        self._X_queue = None
        self._y_queue = None
        self._is_initialized = False
        self.counters_minority = {}  # dictionary --> key : class, value : counts of examples that are of this class/label

    def configure(self):
        self._X_queue = np.zeros((0, self._n_features))
        self._y_queue = np.zeros((0, self._n_targets))
        self._is_initialized = True
        self.counters_minority = self._n_targets # TODO : initialze a dictionary tht contains as keys all classes
                                                #   all classes are potential minority classes

    def add_sample(self, X, y):
        """ Add a (single) sample to the sample window.

        X: numpy.ndarray of shape (1, n_features)
            1D-array of feature for a single sample.

        y: numpy.ndarray of shape (1, n_targets)
            1D-array of targets for a single sample.

        Raises
        ------
        ValueError: If at any moment, a sample with a different number of \
        attributes than that of the n_attributes parameter is passed, a \
        ValueError is raised.

        TypeError: If the buffer type is altered by the user, or isn't \
        correctly initialized, a TypeError may be raised.

        """

        if not self._is_initialized:
            self._n_features = get_dimensions(X)[1]
            self._n_targets = get_dimensions(y)[1]
            self.configure()

        if self._n_features != get_dimensions(X)[1]:
            raise ValueError("Inconsistent number of features in X: {}, previously observed {}.".
                             format(get_dimensions(X)[1], self._n_features))

        if self.size == self.window_size:
            # deduct the example of the specific minority class using the appropriate counter of the class
            self.counters_minority[self._y_queue[0]] -= 1
            # Delete oldest sample
            self._X_queue = np.delete(self._X_queue, 0, axis=0)
            self._y_queue = np.delete(self._y_queue, 0, axis=0)

        self._X_queue = np.vstack((self._X_queue, X))
        self._y_queue = np.vstack((self._y_queue, y))

        # add the new label of the

    # TODO : do the below after a specific amount of instances passed in the window (user specified)
        # TODO : find minority classes : minority_classes = find_minority_classes(_y_queue) : count labels
        # if y belongs in minority classes:
        # counter_minority{y} += 1
        # if counter_minority <= 0.4 len() :
        # then data inside the window is imbalanced so:
        # perform dir


    def delete_oldest_sample(self):
        """ Delete the oldest sample in the window. """
        if self.size > 0:
            self._X_queue = self._X_queue[1:, :]
            self._y_queue = self._y_queue[1:, :]

    def reset(self):
        """ Reset the sliding window. """
        self._n_features = -1
        self._n_targets = -1
        self._X_queue = None
        self._y_queue = None
        self._is_initialized = False

    @property
    def features_buffer(self):
        """ Get the features buffer.

        The shape of the buffer is (window_size, n_features).
        """
        return self._X_queue

    @property
    def targets_buffer(self):
        """ Get the targets buffer

        The shape of the buffer is (window_size, n_targets).
        """
        return self._y_queue

    @property
    def n_targets(self):
        """ Get the number of targets. """
        return self._n_targets

    @property
    def n_features(self):
        """ Get the number of features. """
        return self._n_features

    @property
    def size(self):
        """ Get the window size. """
        return 0 if self._X_queue is None else self._X_queue.shape[0]




