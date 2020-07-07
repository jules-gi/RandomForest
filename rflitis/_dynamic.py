"""
This module provides the Dynamic Random Forest method for decision making.
Based on Scikit-Learn architecture.

Reference :
    Simon Bernard, Sébastien Adam, Laurent Heutte. Dynamic Random Forests.
    Pattern Recognition Letters, Elsevier, 2012, 33 (12), pp.1580-1586.
    10.1016/j.patrec.2012.04.003 . hal-00710083

Contributors :
    Jules Girard, jules.girard@outlook.com
    Simon Bernard, simon.bernard@univ-rouen.fr

Update :    April 3, 2020
"""

import numpy as np

from operator import itemgetter
from sklearn.tree import DecisionTreeClassifier

from rflitis.utils.samples import get_bootstrap_indices, get_class_dictionary
from rflitis.utils.voting import count_votes


class DynamicRandomForestClassifier:
    """
    A Dynamic Random Forest Classifier using sklearn DecisionTreeClassifier.

    ----------
    Parameters
    ----------

    n_estimators: int or 'auto' (default = 'auto')
        The number of trees in the forest.
        If 'auto', the forest sum the k-1 previous OOB error differences, and
        stop growing when this criterion is less than threshold t.
        - sum(diff(error_OOB[-k:])) < t

    sliding_window: tuple of (int, float) or 'auto' (default = 'auto')
        - Positive integer, the number of previous trees (k) used to compute
          OOB error differences. Refers to the sliding window width (default = 20)
        - Positive float, refers to the threshold (t) (default = .01)

    criterion: 'gini' or 'entropy', (default = 'gini')
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split (from sklearn).

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

    max_features: int, float or {'auto', 'sqrt', 'log2'} (default = 'sqrt')
        The number of features to consider when looking for the best split
        (from sklearn).

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

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

    class_weight: dict, list of dict or "balanced" (default = None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of _y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of _y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(_y))``
        For multi-output, the weights of each column of _y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified (from sklearn).

    random_state : int (default = None)
        Set random state


    ---------
    Reference
    ---------

    Simon Bernard, Sébastien Adam, Laurent Heutte. Dynamic Random Forests.
    Pattern Recognition Letters, Elsevier, 2012, 33 (12), pp.1580-1586.
    10.1016/j.patrec.2012.04.003 . hal-00710083
    """

    def __init__(self,
                 n_estimators='auto',
                 sliding_window='auto',
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features='sqrt',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 class_weight=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.random_state = random_state

        self.class_dict_ = None
        self.n_classes_ = None
        self.forest = None
        self.score_oob = None

        if n_estimators == 'auto':
            if sliding_window == 'auto':
                self.sliding_window = (20, .01)
            else:
                if isinstance(sliding_window, tuple):
                    if (len(sliding_window) == 2 and
                            isinstance(sliding_window[0], int) and
                            isinstance(sliding_window[1], float)):
                        if sliding_window[1] >= 0:
                            self.sliding_window = sliding_window
                        else:
                            raise ValueError(
                                'sliding_window width must be >= 0.')
                    else:
                        raise TypeError(
                            'sliding_window must be a tuple of length 2, '
                            'containing an integer and a float.')
                else:
                    raise TypeError(
                        'sliding_window must be a tuple of length 2.')
        else:
            if isinstance(n_estimators, int):
                self.n_estimators = n_estimators
                self.sliding_window = (0, 0)
            else:
                raise TypeError('n_estimators must be an integer.')

    def fit(self, X, y):
        """
        Build a dynamic forest of trees from the training set (_X, _y).
        For the first tree, all instances have the same weight:
        1/n_instances. Then weights are calculated for instances that
        are not known by at least one tree in the forest. Instances
        known by all the previous trees keep the weight 1/n_instances.

        ----------
        Parameters
        ----------
        X : array-like of shape (n_instances, n_features)
            The training input samples.

        y : array-like of shape (n_instances,) or (n_instances, n_outputs)
            The target values (class labels).

        -------
        Returns
        -------
        self : object
        """
        n_samples, n_features = X.shape
        self.class_dict_, self.n_classes_ = get_class_dictionary(y)
        y_values = np.array(itemgetter(*y)(self.class_dict_))
        sample_weights = np.repeat(1 / n_samples, n_samples)
        votes_w = np.zeros((n_samples, self.n_classes_), int)
        votes_oob = np.copy(votes_w)
        instances_oob = np.argwhere(np.sum(votes_oob, axis=1) != 0).T[0]

        self.score_oob = []  # useful?? never used outside this function
        self.forest = []

        crit = 1.
        i = 0
        np.random.seed(self.random_state)
        while crit > self.sliding_window[1]:

            # Weighted sampling
            bootstrap, oob = get_bootstrap_indices(n_samples, n_samples,
                                                   weights=sample_weights)

            # Add a tree to forest
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                class_weight=self.class_weight,
                random_state=self.random_state
            ).fit(X[bootstrap], y[bootstrap],
                  sample_weight=sample_weights[bootstrap])
            self.forest.append(tree)
            i += 1

            # Prediction of this new tree
            predict = np.array(itemgetter(*tree.predict(X))(self.class_dict_))
            votes_w[range(n_samples), predict] += 1
            votes_oob[oob, predict[oob]] += 1

            # Compute weight for instances who were at least OOB once
            if len(instances_oob) != n_samples:
                instances_oob = np.argwhere(np.sum(votes_oob, axis=1) != 0).T[
                    0]
            sample_weights = 1 - (
                        votes_w[range(n_samples), y_values] / np.sum(votes_w,
                                                                     axis=1))
            sample_weights = sample_weights / np.sum(sample_weights)

            # Compute OOB error
            self.score_oob.append(
                np.mean(
                    votes_oob[instances_oob, y_values[instances_oob]] / np.sum(
                        votes_oob[instances_oob], axis=1))
            )

            # Stop criterion
            if self.n_estimators == 'auto' and i >= self.sliding_window[
                0] + np.argmax(self.score_oob):
                crit = -np.sum(
                    np.diff(self.score_oob[-self.sliding_window[0]:]))

            elif i == self.n_estimators:
                break

        self.n_estimators = i

    def predict_proba(self, X):
        """
        Compute labels probability for instances from _X.

        Label probability is computed by the ratio of number of trees in
        the forest that predict a label on the total number of trees in
        the forest.

        ----------
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        -------
        Returns
        -------
        prob_y : array-like of shape (n_samples, n_labels)
            The labels probability of instances.
        """
        n_output = X.shape[0]
        if self.forest is not None:
            votes = np.asarray([tree.predict(X) for tree in self.forest]).T
            votes = list(map(count_votes, votes))
            prob_y = np.zeros((n_output, self.n_classes_))
            for i in range(n_output):
                prob_y[i, itemgetter(*votes[i][0])(self.class_dict_)] = \
                votes[i][1]

            prob_y /= self.n_estimators
        else:
            raise ValueError("Fit data before predict.")

        return prob_y

    def predict(self, X):
        """
        Predict labels for _X.

        The predicted label is the one with highest probability estimate
        across the trees.

        ----------
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        -------
        Returns
        -------
        prob_y : array-like of shape (n_samples,)
            The labels prediction of instances.
        """
        if self.forest is not None:
            prob_y = self.predict_proba(X)
            votes = np.asarray([*self.class_dict_.keys()])[
                np.argmax(prob_y, axis=1)]
        else:
            raise ValueError("Fit data before predict.")

        return votes

    def score(self, X, y, monitoring=False):
        """
        Compute the forest prediction score.

        The forest prediction score is a ratio between the number of
        input well classified by the forest and the total number of
        inputs.

        ----------
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The input labels.
        monitoring : boolean (default=False)
            If True, return the forest's score for each tree added in
            the forest.

        -------
        Returns
        -------
        score : float (0 <= score <= 1)
            The forest prediction score.

        score_list : list of len(n_estimators) (if monitoring=True)
            The forest prediction score for every trees added in the
            forest.
        """
        n_output = X.shape[0]
        if self.forest is not None:
            if monitoring:
                score_list = []
                for k in np.arange(1, self.n_estimators + 1):
                    votes = np.asarray(
                        [tree.predict(X) for tree in self.forest[:k]]).T
                    votes = list(map(count_votes, votes))
                    prob_y = np.zeros((n_output, self.n_classes_))
                    for i in range(n_output):
                        prob_y[i, itemgetter(*votes[i][0])(self.class_dict_)] = \
                        votes[i][1]
                    prob_y /= k

                    predict = np.asarray([*self.class_dict_.keys()])[
                        np.argmax(prob_y, axis=1)]
                    score_list.append(np.sum(y == predict) / n_output)
                score = score_list[-1]

                return score, score_list

            else:
                predict = self.predict(X)
                score = np.sum(y == predict) / n_output

                return score
        else:
            raise ValueError("Fit data before predict.")
