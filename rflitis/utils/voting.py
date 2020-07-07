"""
This module provides various utilities for voting methods

Contributors :
    Jules Girard, jules.girard@outlook.com

Update :    January 24, 2020
"""

import numpy as np


def count_votes(votes):
    """
    Count labels votes from a tree prediction,
    from an array-like of shape (n_instances,)

    ----------
    Parameters
    ----------

    votes : array-like of shape (n_instances,) (require)
        Labels prediction from a tree _classifier.

    -------
    Returns
    -------

    count : array-like of shape (n_labels, 2)
        Number of instances assigned to differents labels.
    """
    count = np.unique(votes, return_counts=True)

    return count
