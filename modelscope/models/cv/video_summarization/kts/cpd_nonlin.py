# Part of the implementation is borrowed and modified from KTS,
# publicly available at https://github.com/TatsuyaShirakawa/KTS

import numpy as np


def calc_scatters(K):
    """Calculate scatter matrix: scatters[i,j] = {scatter of the sequence with
    starting frame i and ending frame j}
    """
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n + 1, n + 1))
    # TODO: use the fact that K - symmetric
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1)

    diagK2 = np.diag(K2)

    i = np.arange(n).reshape((-1, 1))
    j = np.arange(n).reshape((1, -1))

    ij_f32 = ((j - i + 1).astype(np.float32) + (j == i - 1).astype(np.float32))
    diagK2_K2 = (
        diagK2[1:].reshape((1, -1)) + diagK2[:-1].reshape(
            (-1, 1)) - K2[1:, :-1].T - K2[:-1, 1:])
    scatters = (
        K1[1:].reshape((1, -1)) - K1[:-1].reshape(
            (-1, 1)) - diagK2_K2 / ij_f32)

    scatters[j < i] = 0

    return scatters


def cpd_nonlin(K,
               ncp,
               lmin=1,
               lmax=100000,
               backtrack=True,
               verbose=True,
               out_scatters=None):
    """Change point detection with dynamic programming

    :param K: Square kernel matrix
    :param ncp: Number of change points to detect (ncp >= 0)
    :param lmin: Minimal length of a segment
    :param lmax: Maximal length of a segment
    :param backtrack: If False - only evaluate objective scores (to save memory)
    :param verbose: If true, print verbose message
    :param out_scatters: Output scatters
    :return: Tuple (cps, obj_vals)
        - cps - detected array of change points: mean is thought to be constant
            on [ cps[i], cps[i+1] )
        - obj_vals - values of the objective function for 0..m changepoints
    """
    m = int(ncp)  # prevent numpy.int64

    n, n1 = K.shape
    assert n == n1, 'Kernel matrix awaited.'
    assert (m + 1) * lmin <= n <= (m + 1) * lmax
    assert 1 <= lmin <= lmax

    if verbose:
        print('Precomputing scatters...')
    J = calc_scatters(K)

    if out_scatters is not None:
        out_scatters[0] = J

    if verbose:
        print('Inferring best change points...')
    # Iden[k, l] - value of the objective for k change-points and l first frames
    Iden = 1e101 * np.ones((m + 1, n + 1))
    Iden[0, lmin:lmax] = J[0, lmin - 1:lmax - 1]

    if backtrack:
        # p[k, l] --- 'previous change' --- best t[k] when t[k+1] equals l
        p = np.zeros((m + 1, n + 1), dtype=int)
    else:
        p = np.zeros((1, 1), dtype=int)

    for k in range(1, m + 1):
        for l_frame in range((k + 1) * lmin, n + 1):
            tmin = max(k * lmin, l_frame - lmax)
            tmax = l_frame - lmin + 1
            c = J[tmin:tmax, l_frame - 1].reshape(-1) + \
                Iden[k - 1, tmin:tmax].reshape(-1)
            Iden[k, l_frame] = np.min(c)
            if backtrack:
                p[k, l_frame] = np.argmin(c) + tmin

    # Collect change points
    cps = np.zeros(m, dtype=int)

    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k - 1] = p[k, cur]
            cur = cps[k - 1]

    scores = Iden[:, n].copy()
    scores[scores > 1e99] = np.inf
    return cps, scores
