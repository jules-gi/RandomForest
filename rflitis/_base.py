import numbers
import numpy as np

from abc import ABCMeta, abstractmethod
from warnings import warn
from sklearn.ensemble.forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from rflitis.utils.samples import get_class_dictionary
from rflitis.utils.voting import count_votes


class BaseRFLitisClassifier(ForestClassifier, metaclass=ABCMeta):
    """
    Base class for forest classifier of LITIS methods, based on scikit-learn.

    Parameters
    ----------
    base_estimator: object
        The base estimator from which the ensemble is built.

    estimator_params: list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    n_estimators: int, default=100
        The number of estimators in the ensemble.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 estimator_params=tuple(),
                 n_estimators=100,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        # self.base_estimator_ = base_estimator
        self.estimators_ = []
        self.n_classes_ = None
        self.class_dict_ = None

    def fit(self, X, y, sample_weight=None):
        self.class_dict_, self.n_classes_ = get_class_dictionary(y)
        super().fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X, voting="soft"):
        """
        DocStrings
        """
        if voting is "soft":
            return super().predict_proba(X)

        if X.ndim == 1:
            X = X.reshape((1, self.n_features_))
            warn("Classifier predicted only 1 sample.")
        n_outputs = X.shape[0]

        if len(self.estimators_) > 0:
            votes = np.asarray([estimator.predict_proba(X)[:np.newaxis]
                                for estimator in self.estimators_]).T
            n_votes = list(map(count_votes, np.argmax(votes, axis=0)))
            prob = np.zeros((n_outputs, self.n_classes_))
            for i in range(n_outputs):
                prob[i, n_votes[i][0]] = n_votes[i][1]
            prob /= len(self.estimators_)
        else:
            raise ValueError("This _classifier is not fitted yet. "
                             "Call 'fit' with appropriate arguments "
                             "before using the _classifier.")
        return prob

    def predict(self, X, voting="soft"):
        """
        DocStrings
        """
        if voting is "soft":
            return super().predict(X)

        if self.class_dict_ is None:
            raise ValueError("This _classifier is not fitted yet. "
                             "Call 'fit' with appropriate arguments "
                             "before using the _classifier.")

        proba = self.predict_proba(X, voting=voting)
        prediction = np.asarray([*self.class_dict_.keys()])[np.argmax(proba, axis=1)]
        return prediction

    def score(self, X, y, voting="soft", sample_weight=None):
        """
        DocStrings
        """
        if voting is "soft":
            return super().score(X, y, sample_weight=sample_weight)

        score = accuracy_score(y, self.predict(X, voting=voting),
                               sample_weight=sample_weight)
        return score
