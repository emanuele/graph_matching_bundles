import numpy as np
from DSPFP import DSPFP_faster_K


def graph_matching(S_A, S_B, distance, alpha=0.5, max_iter1=100,
                   max_iter2=100, initialization='NN',
                   similarity='exp(-x)', epsilon=1.0e-8, verbose=True):
    """Wrapper of the DSPFP algorithm to deal with streamlines. In
    addition to calling DSPFP, this function adds initializations of
    the graph matching algorithm that are meaningful for streamlines,
    as well as some (optional) conversions from distances to
    similarities.
    """
    assert(len(S_B) >= len(S_A))  # required by DSPFP
    if verbose:
        print("Computing graph matching between streamlines.")
        print("Computing the distance matrix between streamlines in each set")

    dm_A = distance(S_A, S_A)
    dm_B = distance(S_B, S_B)
    K = distance(S_A, S_B)

    # Notice that the initialization transposes the matrix because the
    # logic of DSPFP is DSPFP(B,A), which is opposite to that of our
    # graph_matching(A,B):
    if initialization == 'NN':
        X_init = K.T
    elif initialization == 'random':
        X_init = np.random.uniform(size=(len(S_A), len(S_B))).T
    else:
        # flat initialization, default of DSPFP
        X_init = None

    # Wheter to use distances or similarities and, in case, which
    # similarity function
    if similarity == '1/x':
        sm_A = 1.0 / (1.0 + dm_A)
        sm_B = 1.0 / (1.0 + dm_B)
        if initialization == 'NN':
            X_init = 1.0 / (1.0 + X_init)

    elif similarity == 'exp(-x)':
        tmp = np.median(dm_A)
        sm_A = np.exp(-dm_A / tmp)
        tmp = np.median(dm_B)
        sm_B = np.exp(-dm_B / tmp)
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
    X = DSPFP_faster_K(sm_B, sm_A, K=K,
                       alpha=alpha,
                       max_iter1=max_iter1,
                       max_iter2=max_iter2,
                       X=X_init, verbose=verbose)

    ga = greedy_assignment(X)
    corresponding_streamlines = ga.argmax(0)
    unassigned = (ga.sum(0) == 0)
    corresponding_streamlines[unassigned] = -1
    return corresponding_streamlines
