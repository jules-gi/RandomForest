"""
This module provides various utilities for samples manipulations

Contributors :
    Jules Girard, jules.girard@outlook.com
    Simon Bernard, simon.bernard@univ-rouen.fr

Update :    April 3, 2020
"""

import numpy as np
import numbers

from sklearn.utils.validation import check_random_state


def get_bootstrap_indices(n_samples,
                          n_samples_bootstrap=None,
                          weights=None,
                          random_state=None):
    """
    Get a bootstrap sample indices and the corresponding out-of-bag indices

    ----------
    Parameters
    ----------

    n_samples : int (require)
        Total number of instances available in the initial sample

    n_samples_bootstrap : int, or None (default=None)
        Number of instances of the bootstrap sample to generate
        if None, n_samples_bootstrap is set to n_samples

    weights : array-like of shape (n_instances,), or None (default=None)
        The sample weights

    random_state : int, or None (default=None)
        Control the randomness of the bootstrapping

    -------
    Returns
    -------

    bootstrap_indices : array-like of shape (n_instances-n_oob,)
        The index of bootstrap instances.

    oob_indices : array-like of shape (n_instances-n_bootstrap,)
        The index of OOB instances.
    """

    if n_samples_bootstrap is None:
        n_samples_bootstrap = n_samples

    rnd = check_random_state(random_state)
    bootstrap_indices = rnd.choice(a=range(n_samples), size=n_samples_bootstrap, replace=True, p=weights)
    oob_indices = np.asarray(range(n_samples))[np.isin(range(n_samples), bootstrap_indices, invert=True)]

    return bootstrap_indices, oob_indices


def get_subspace_indices(n_features,
                         n_features_subspace,
                         weights=None,
                         random_state=None):
    """
    Create a weighted bootstrap and its OOB.

    ----------
    Parameters
    ----------

    n_features : int (require)
        Total number of features available in the initial sample

    n_features_subspace : int (require)
        Number of feature to sample to form the random subspace

    weights : array-like of shape (n_instances,), or None (default=None)
        The feature weights

    random_state : int, or None (default=None)
        Control the randomness

    -------
    Returns
    -------

    subpace_indices : array-like of shape (n_instances-n_oob,)
        The index of the features in the random subspace.
    """

    rnd = check_random_state(random_state)
    subpace_indices = rnd.choice(a=range(n_features), size=n_features_subspace, replace=False, p=weights)
    return subpace_indices


def get_class_dictionary(y):
    """
    Get a class dictionary from a vector of labels

    ----------
    Parameters
    ----------

    y : array-like of shape (n_instances,) (require)
        vector of labels from which to build the class dictionary

    -------
    Returns
    -------

    class_dict : dict (keys = labels, values = unique_integer)
        Class dictionary that map the class name to its unique integer

    n_classes : int
        Number of labels
    """

    class_dict = {}
    i = 0
    for label in np.unique(y):
        class_dict[label] = i
        i += 1
    n_classes = len(class_dict)

    return class_dict, n_classes
