# Implementation of Adaline Gradient Descend
import numpy as np


class AdalineGD(object):
    """Adaptive Linear Neuron classifier.
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    random_state : int
    Random number generator seed for random weight
    initialization.
    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    cost_ : list
    Sum-of-squares cost function value in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = []
        self.cost_ = []

    def net_input(self, features):
        """Calculate net input
        Example
        -------
        arr1 = np.array([[1, 2, 3], [2, 3, 4]])
        arr2 = np.array([.2, .5, .7])
        np.dot(arr1, arr2)
        will return array([3.3, 4.7]) object
        which is equivalent to w (arr2) transpose X (arr1)
        """
        return np.dot(features, self.w_[1:]) + self.w_[0]

    @staticmethod
    def activation(z):
        """Compute the linear activation"""
        return z

    def predict(self, features):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(features)) >= 0.0, 1, -1)

    def fit(self, features, targets):
        """ Fit training data.
        Parameters
        ----------
        features : {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of
        samples and
        n_features is the number of features.
        targets : array-like, shape = [n_samples]
        Target values.
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + features.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            # take the whole batch of X and calculate Y hat
            net_input = self.net_input(features)
            output = self.activation(net_input)
            # calculate the errors
            errors = (targets - output)
            # weights update
            # features T will be a m by n and errors will by n by 1
            # so the weights update will be m by 1
            self.w_[1:] += self.eta * features.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # calculate costs
            cost = (errors**2).sum() / 2
            self.cost_.append(cost)
        return self


class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    shuffle : bool (default: True)
    Shuffles training data every epoch if True
    to prevent cycles.
    random_state : int
    Random number generator seed for random weight
    initialization.

    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    cost_ : list
    Sum-of-squares cost function value averaged over all
    training samples in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        self.cost_ = None

    def _initialize_weight(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)

    def _shuffle(self, features, targets):
        """Shuffle training data"""
        r = self.rgen.permutation(len(targets))
        return features[r], targets[r]

    def net_input(self, features):
        """Calculate net input
        Example
        -------
        arr1 = np.array([[1, 2, 3], [2, 3, 4]])
        arr2 = np.array([.2, .5, .7])
        np.dot(arr1, arr2)
        will return array([3.3, 4.7]) object
        which is equivalent to w (arr2) transpose X (arr1)
        """
        return np.dot(features, self.w_[1:]) + self.w_[0]

    @staticmethod
    def activation(z):
        """Compute the linear activation"""
        return z

    def predict(self, features):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(features)) >= 0.0, 1, -1)

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def fit(self, features, targets):
        """ Fit training data.
        Parameters
        ----------
        features : {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and
        n_features is the number of features.
        targets : array-like, shape = [n_samples]
        Target values.
        Returns
        -------
        self : object
        """
        self._initialize_weight(features.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                features, targets = self._shuffle(features, targets)
            cost = []
            for xi, yi in zip(features, targets):
                # in that case xi is a row vector [xi_1, xi_2] and yi is a scalar
                # so the dot will be [xi_1yi, xi_2yi]
                cost.append(self._update_weights(xi, yi))
            avg_cost = sum(cost) / len(targets)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, features, targets):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weight(features.shape[1])
        if targets.ravel().shape[0] > 1:
            for xi, yi in zip(features, targets):
                # in that case xi is a row vector [xi_1, xi_2] and yi is a scalar
                # so the dot will be [xi_1yi, xi_2yi]
                self._update_weights(xi, yi)
        else:
            self._update_weights(features, targets)
        return self
