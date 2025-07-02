# import math
#
#
# def do_naive_bayes_prediction(X, observed_class_distribution: dict, attribute_observers: dict):
#     """
#     Perform Naive Bayes prediction
#
#     Parameters
#     ----------
#     X: numpy.ndarray, shape (1, n_features)
#         A feature's vector.
#
#     observed_class_distribution: dict
#         Observed class distribution
#
#     attribute_observers: dict
#         Attribute (features) observer
#
#     Returns
#     -------
#     votes
#         dict
#
#     Notes
#     -----
#     This method is not intended to be used as a stand-alone method.
#     """
#     observed_class_sum = sum(observed_class_distribution.values())
#     if observed_class_distribution == {} or observed_class_sum == 0.0:
#         # No observed class distributions, all classes equal
#         return {0: 0.0}
#     votes = {}
#     for class_index, observed_class_val in observed_class_distribution.items():
#         votes[class_index] = observed_class_val / observed_class_sum
#         if attribute_observers:
#             for att_idx in range(len(X)):
#                 if att_idx in attribute_observers:
#                     obs = attribute_observers[att_idx]
#                     tmp = votes[class_index] * obs.probability_of_attribute_value_given_class(
#                         X[att_idx], class_index)
#                     votes[class_index] = tmp if not math.isnan(tmp) else 0
#     return votes


#Marina implementation


import math
import numpy as np


def safe_probability(prob):
    """ Prevent underflow/overflow by clipping extreme probability values """
    return np.clip(prob, 1e-10, 1e10)


def safe_log(x):
    """ Compute log safely, avoiding log(0) and underflow issues """
    return np.log(np.maximum(x, 1e-10))


def safe_divide(numerator, denominator):
    """ Prevent division by zero by ensuring denominator is non-zero """
    return numerator / np.maximum(denominator, 1e-10)


def do_naive_bayes_prediction(X, observed_class_distribution: dict, attribute_observers: dict):
    """
    Perform Naive Bayes prediction with numerical stability.

    Parameters
    ----------
    X: numpy.ndarray, shape (1, n_features)
        A feature's vector.

    observed_class_distribution: dict
        Observed class distribution

    attribute_observers: dict
        Attribute (features) observer

    Returns
    -------
    votes
        dict

    Notes
    -----
    This method is not intended to be used as a stand-alone method.
    """
    observed_class_sum = sum(observed_class_distribution.values())

    #  Avoid division by zero
    if not observed_class_distribution or observed_class_sum == 0.0:
        return {0: 0.0}

    votes = {}

    for class_index, observed_class_val in observed_class_distribution.items():
        #  Safe division to prevent zero division
        votes[class_index] = safe_divide(observed_class_val, observed_class_sum)

        if attribute_observers:
            for att_idx in range(len(X)):
                if att_idx in attribute_observers:
                    obs = attribute_observers[att_idx]

                    #  Safe probability retrieval
                    prob = safe_probability(obs.probability_of_attribute_value_given_class(X[att_idx], class_index))

                    #  Ensure votes and probabilities are in a stable range
                    safe_vote = safe_probability(votes[class_index])
                    tmp = safe_vote * prob  # Multiplication

                    #  Prevent NaN values
                    votes[class_index] = tmp if not math.isnan(tmp) else 0

    return votes
