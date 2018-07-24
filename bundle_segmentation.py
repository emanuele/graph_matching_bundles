from __future__ import division, print_function, absolute_import
import numpy as np
from DSPFP import DSPFP_faster_K, greedy_assignment
from linear_assignment import LinearAssignment


def graph_matching(S_A, S_B, distance, lam=0.5, alpha=0.5,
                   max_iter1=100, max_iter2=100, initialization='NN',
                   similarity='exp(-x)', epsilon=1.0e-8, verbose=True,
                   LAPJV=True):
    """Wrapper of the DSPFP algorithm to deal with streamlines. In
    addition to calling DSPFP, this function adds initializations of
    the graph matching algorithm that are meaningful for streamlines,
    as well as some (optional) conversions from distances to
    similarities.
    """
    assert(len(S_B) >= len(S_A))  # required by DSPFP
    if verbose:
        print("Computing graph matching between streamlines.")
        print("Computing the distance matrix between streamlines in each set, for QAP")

    dm_A = distance(S_A, S_A)
    dm_B = distance(S_B, S_B)

    if verbose:
        print("Computing the distance (cost) matrix between the streamlines of the two sets, for LAP")

    # K is the 'benefit' matrix, i.e. the opposite of the cost matrix,
    # for the linear assignment part:
    K = -distance(S_A, S_B)

    if verbose:
        print('Initialization: %s' % initialization)

    if initialization == 'NN':
        X_init = K / K.max()
    elif initialization == 'random':
        X_init = np.random.uniform(size=(len(S_A), len(S_B)))
    else:
        # flat initialization, default of DSPFP
        X_init = None

    # Wheter to use distances or similarities and, in case, which
    # similarity function
    if verbose:
        print('Similarity: %s' % similarity)

    if similarity == '1/x':
        sm_A = 1.0 / (1.0 + dm_A)
        sm_B = 1.0 / (1.0 + dm_B)
        K = -(1.0 / (1.0 + K))
        if initialization == 'NN':
            X_init = 1.0 / (1.0 + X_init)

    elif similarity == 'exp(-x)':
        tmp = np.median(dm_A)
        sm_A = np.exp(-dm_A / tmp)
        # tmp = np.median(dm_B)
        sm_B = np.exp(-dm_B / tmp)
        # tmp = np.median(-K)
        K = np.exp(K / tmp)
        if initialization == 'NN':
            tmp = np.median(X_init)
            X_init = np.exp(-X_init / tmp)

    else:  # Don't use similarity
        sm_A = dm_A
        sm_B = dm_B
        if initialization == 'NN':
            X_init = 1.0 / (1.0 + X_init)  # anyway X_init needs
                                           # similarity when usign NN
                                           # initialization

    if verbose:
        print("Computing graph-matching via DSPFP")

    # We perform DSPFP(B,A) and not DSPFP(A,B), because the original
    # algorithm has the opposite logic of what we need (see the
    # paper):
    X = DSPFP_faster_K(sm_B, sm_A, K=K.T, lam=lam, alpha=alpha,
                       max_iter1=max_iter1, max_iter2=max_iter2,
                       X=np.transpose(X_init), verbose=verbose)

    if LAPJV:
        if verbose:
            print('Finding the exact solution of the LAP with LAPJV.')

        corresponding_streamlines = LinearAssignment(-X.T).solution
    else:
        if verbose:
            print('Finding the greedy solution of the LAP.')

        ga = greedy_assignment(X)
        corresponding_streamlines = ga.argmax(0)
        unassigned = (ga.sum(0) == 0)
        corresponding_streamlines[unassigned] = -1

    return corresponding_streamlines


if __name__ == '__main__':
    np.random.seed(0)

    n_A = 100 # 1000
    n_B = 100 # 10000
    d = 2
    lam = 1.0

    S_A = np.random.uniform(size=(n_A, d))
    S_B = S_A + np.random.normal(size=(n_A, d)) * 0.01
    S_B = np.vstack([S_B, np.random.uniform(size=(n_B - n_A, d))])

    from functools import partial
    from distances import euclidean_distance, parallel_distance_computation
    distance = partial(parallel_distance_computation, distance=euclidean_distance)

    corresponding_streamlines = graph_matching(S_A, S_B, distance,
                                               lam=lam, alpha=0.5,
                                               max_iter1=100,
                                               max_iter2=100,
                                               initialization='NN',
                                               similarity='exp(-x)',
                                               epsilon=1.0e-8,
                                               verbose=True,
                                               LAPJV=True)

    import matplotlib.pyplot as plt
    plt.interactive(True)
    plt.figure()
    plt.plot(S_A[:, 0], S_A[:, 1], 'ro')
    for i in range(n_A):
        plt.arrow(S_A[i, 0], S_A[i, 1], S_B[corresponding_streamlines[i], 0] - S_A[i, 0], S_B[corresponding_streamlines[i], 1] - S_A[i, 1], head_width=0.01, head_length=0.01, fc='g', ec='g', length_includes_head=True)

    plt.plot(S_B[:, 0], S_B[:, 1], 'bo')
