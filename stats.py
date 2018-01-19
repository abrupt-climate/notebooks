import numpy as np


def weighted_quartiles(sample, weights):
    """Compute the minimum, first quartile, median, third
    quartile and the maximum of the weighted sample.

    :param sample: data sample, one-dimensional array
    :param weights: weights, one-dimensional array, same length as sample
    :returns: an array of five elements [min, 1st, med, 3rd, max]
    """
    quartiles = np.array([0, 1/4, 1/2, 3/4, 1])
    order = np.argsort(sample)
    F = np.cumsum(weights[order])
    indices = [min(i, order.size-1)
               for i in np.searchsorted(F, F[-1] * quartiles)]
    return sample[order[indices]]
