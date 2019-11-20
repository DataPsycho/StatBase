from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


class SBS:
    """Sequential backward selection class
    :parameter
    estimator: the model to fit
    k_features: the number of feature to test
    test_size: size of test sample
    random_state: selection of random seed
    """
    def __init__(self,
                 estimator,
                 k_features,
                 scoring=accuracy_score,
                 test_size=0.25,
                 random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        self.indices_ = None
        self.subsets_ = None
        self.score_ = None
        self.k_score_ = None

    def transform(self, features):
        """Get final transformed feature set after finish benchmarking"""
        return features[:, self.indices_]

    def _calc_score(self, feat_train, tar_train, feat_test, tar_test, indices):
        """Fit a model and calculate scores for a given index list"""
        self.estimator.fit(feat_train[:, indices], tar_train)
        predictions = self.estimator.predict(feat_test[:, indices])
        score = self.scoring(tar_test, predictions)
        return score

    def fit(self, features, targets):
        """
        Sequentially fit models with based on a reduced feature list
        Iteration example: For example if we start with 5 feature list and we want to
        reduce to 3 at the end. Then in the first step we will fit for 5 features and go
        to the while loop to check if 5 > 3 then in the combination we fit for 4 feature.
        End of the while loop we reduce the feature to 5-1 = 4 and check again if 4 > 3.
        Then we fit models again for all possible combination of 3's and reduce the feature
        to 4 -1 = 3. Here while loop will stop execution.
        :parameter
        features: feature data array (X)
        targets: target data array (y)
        """
        feat_train, feat_test, tar_train, tar_test = (
            train_test_split(
                features,
                targets,
                test_size=self.test_size,
                random_state=self.random_state
            )
        )
        dim = feat_train.shape[1]  # No of features
        self.indices_ = tuple(range(dim))  # index the feature from 0 to D
        self.subsets_ = [self.indices_]
        # fit a initial model with the specific index list
        score = self._calc_score(feat_train, tar_train, feat_test, tar_test, self.indices_)
        # store the initial score in to list
        self.score_ = [score]

        # Calculate scores until the sub feature list has been reach
        while dim > self.k_features:
            scores = []
            subsets = []
            # create all possible combination of features as p (1, 2, 4), (1, 3, 4)
            # with the r values per combination
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(feat_train, tar_train, feat_test, tar_test, p)
                scores.append(score)
                subsets.append(p)

            # find the position of the best score
            best = np.argmax(scores)
            # find the index list of the best score
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            # Reduce feature index list by 1
            dim -= 1

            self.score_.append(scores[best])
        # keep the last core
        self.k_score_ = self.score_[-1]
