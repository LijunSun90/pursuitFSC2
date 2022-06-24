import numpy as np
import scipy.signal


def combined_shape(n_agents, length, shape=None):
    if shape is None:
        return length,

    return (n_agents, length, shape) if np.isscalar(shape) else (n_agents, length, *shape)


def discount_cumulative_sum(x, discount):
    """
    Magic from rllab for computing discounted cumulative sums of vectors.
    Input:
        Vector x =
        [x0,
         x1,
         x2]
    Output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    # scipy.signal.lfilter(b, a, x, axis=-1, zi=None)
    # scipy.signal.lfilter([1], [1, -0.9], [1, 2, 3])
    # array([1.  , 2.9 , 5.61])
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

