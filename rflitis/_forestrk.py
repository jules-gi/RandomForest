"""
This module provides the ForestRK method for decision making.
Based on Scikit-Learn architecture.

Reference :
    Bernard, S., Heutte, L., Adam, S., "Forest-R.K.: A New Random Forest
    Induction Method", Fourth International Conference on Intelligent Computing
    (ICIC), Lecture Notes in Computer Science, vol 5227, pp.430-437, 2008.

Contributors :
    Jules Girard, jules.girard@outlook.com
    Simon Bernard, simon.bernard@univ-rouen.fr

Update :  april 30, 2020
"""

import numpy as np
import copy
from warnings import warn

from sklearn.utils.validation import check_random_state
from sklearn.tree import DecisionTreeClassifier
from rflitis._base import BaseRFLitisClassifier
from rflitis.utils.samples import get_bootstrap_indices, get_class_dictionary

MAX_INT = np.iinfo(np.int32).max


class ForestRKClassifier(BaseRFLitisClassifier):
    """
    A Forest-RK Classifier using sklearn DecisionTreeClassifier.

    ----------
    Parameters
    ----------

    n_estimators: int or 'auto' (default = 'auto')
        The number of trees in the forest.
        If 'auto', the forest sum the k-1 previous OOB error differences, and
        stop growing when this criterion is less than threshold t.
        - sum(diff(error_OOB[-k:])) < t

    criterion: {'gini', 'entropy'}, (default = 'gini')
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split (from sklearn).

    max_features : {"auto", "sqrt", "log2", "random"}, int or float, default="random"
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If "random", then `max_features` is set to a random int for each tree
        of the forest.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth: int (default = None)
        The maximum depth of trees. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples (from sklearn).

    min_samples_split: int or float (default = 2)
        The minimum number of samples required to split an internal node
        (from sklearn).
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf: int or float (default = 1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression (from sklearn).
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf: float (default = .0)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided (from sklearn).

    max_leaf_nodes: int (default = None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes (from sklearn).

    min_impurity_decrease: float (default = .0)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following:
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed (from sklearn).

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree (from sklearn).

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization accuracy (from sklearn).

    random_state : int or RandomState, default=None
        Controls the randomness of the bootstrapping of the samples used
        when building trees (if bootstrap=True), the number of features,
        and the sampling of the features to consider when looking for the best
        split at each node (if max_features < n_features).

    ---------
    Reference
    ---------

    Bernard, S., Heutte, L., Adam, S., "Forest-R.K.: A New Random Forest
    Induction Method", Fourth International Conference on Intelligent Computing
    (ICIC), Lecture Notes in Computer Science, vol 5227, pp.430-437, 2008.
    """

    def __init__(self,
                 n_estimators=100,
                 warm_start=False,
                 criterion="gini",
                 max_features="random",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 bootstrap=True,
                 oob_score=False,
                 random_state=None):
        super().__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=None)

        # used in _make_estimator:
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.random_state = random_state
        self.max_features = max_features
        self._max_features = max_features

        self.n_outputs_ = 1         # multiple outputs not supported
        self.classes_ = None        # ndarray of the available classes
        self.n_classes_ = None      # number of classes
        self.n_samples_ = None      # number of instances
        self.n_features_ = None     # number of features
        self.class_dict_ = None     # class dictionnary as returned by .utils.samples.get_class_dictionary()
        self.oob_indices_ = []

    def fit(self, X, y, sample_weight=None):
        """
        This method build a forest from training set (X, y)
        """
        if self.max_features != "random":
            super(ForestRKClassifier, self).fit(X, y, sample_weight)

        else:

            if not self.bootstrap and self.oob_score:
                raise ValueError("Out of bag estimation only available"
                                 " if bootstrap=True")

            # init. required for the Scikit-learn _make_estimators functions to work properly
            #   taken from BaseForest.fit and ForestClassifier._validate_y_class_weight(self, _y)
            #   but simplified since multiple outputs and class weights are not supported here
            self.class_dict_, self.n_classes_ = get_class_dictionary(y)
            self.n_samples_, self.n_features_ = X.shape
            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
            self._validate_estimator()

            # init. three different random generator for independence of all the randomization processes
            random_state_estimator = check_random_state(self.random_state)
            random_state_sample = copy.deepcopy(random_state_estimator)
            random_state_max_features = copy.deepcopy(random_state_estimator)

            n_more_estimators = self.n_estimators - len(self.estimators_)
            if n_more_estimators < 0:
                raise ValueError(f"n_estimators={self.n_estimators} must be "
                                 f"larger or equal to len(estimators_)="
                                 f"{len(self.estimators_)} when "
                                 f"warm_start==True.")
            elif n_more_estimators == 0:
                warn("Warm-start fitting without increasing n_estimators "
                     "does not fit new trees.")
            else:

                if self.warm_start and len(self.estimators_) > 0:
                    random_state_estimator.randint(MAX_INT, size=len(self.estimators_))
                    random_state_max_features.randint(MAX_INT, size=len(self.estimators_))
                    for i in range(len(self.estimators_)):
                        get_bootstrap_indices(self.n_samples_,
                                              random_state=random_state_sample)

                for i in range(n_more_estimators):
                    self.max_features = random_state_max_features.randint(1, self.n_features_)
                    tree = self._make_estimator(random_state=random_state_estimator,
                                                append=False)
                    if self.bootstrap:
                        bootstrap, oob = get_bootstrap_indices(self.n_samples_, random_state=random_state_sample)
                        tree.fit(X[bootstrap], y[bootstrap])
                        self.oob_indices_.append(oob)
                    else:
                        tree.fit(X, y)
                    self.estimators_.append(tree)

            if self.oob_score:
                self._set_oob_score(X, y)
                # TODO : OOB score not supported


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    random_state = 42
    X, y = fetch_openml("madelon", version=1, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33333,
                                                        random_state=random_state)

    clf_1 = ForestRKClassifier(n_estimators=15,
                               max_features='random',
                               random_state=random_state)
    clf_1.fit(X_train, y_train)
    print(clf_1.score(X_test, y_test))

    clf_2 = ForestRKClassifier(n_estimators=5,
                               warm_start=True,
                               max_features='random',
                               random_state=random_state)
    clf_2.fit(X_train, y_train)
    clf_2.n_estimators += 5
    clf_2.fit(X_train, y_train)
    clf_2.n_estimators += 5
    clf_2.fit(X_train, y_train)

    print(clf_2.score(X_test, y_test))
    print("Finished")
