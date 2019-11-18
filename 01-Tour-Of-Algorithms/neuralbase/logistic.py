import numpy as np


class LogisticRegressionGD(object):
    """Logistic Regression Classifier using gradient descent.
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
    Logistic cost function value in each epoch.
    """
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = []
        self.cost_ = []

    def net_input(self, features):
        """Calculate net input"""
        return np.dot(features, self.w_[1:]) + self.w_[0]

    @staticmethod
    def activation(z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, features):
        """Return class label after unit step"""
        return np.where(self.net_input(features) >= 0.0, 1, 0)

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
            net_input = self.net_input(features)
            output = self.activation(net_input)
            errors = (targets - output)
            self.w_[1:] += self.eta * features.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # calculating the logistic cost
            cost = (
                -targets.dot(np.log(output)) - ((1 - targets).dot(np.log(1 - output)))
            )
            self.cost_.append(cost)
        return self


