# Part of the implementation is borrowed and modified from KTS,
# publicly available at https://github.com/TatsuyaShirakawa/KTS

import numpy as np

from .cpd_nonlin import cpd_nonlin


def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """Detect change points automatically selecting their number

    :param K: Kernel between each pair of frames in video
    :param ncp: Maximum number of change points
    :param vmax: Special parameter
    :param desc_rate: Rate of descriptor sampling, vmax always corresponds to 1x
    :param kwargs: Extra parameters for ``cpd_nonlin``
    :return: Tuple (cps, costs)
        - cps - best selected change-points
        - costs - costs for 0,1,2,...,m change-points
    """
    m = ncp
    _, scores = cpd_nonlin(K, m, backtrack=False, **kwargs)

    N = K.shape[0]
    N2 = N * desc_rate  # length of the video before down-sampling

    penalties = np.zeros(m + 1)
    # Prevent division by zero (in case of 0 changes)
    ncp = np.arange(1, m + 1)
    penalties[1:] = (vmax * ncp / (2.0 * N2)) * (np.log(float(N2) / ncp) + 1)

    costs = scores / float(N) + penalties
    m_best = np.argmin(costs)
    cps, scores2 = cpd_nonlin(K, m_best, **kwargs)

    return cps, scores2
