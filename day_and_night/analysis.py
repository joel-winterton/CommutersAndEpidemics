"""
Functions to make analysis tidier
"""
import numpy as np
from copy import deepcopy
from model import simulate


def filter_population(s, i):
    """"
    Remove non-commuters from simulation data and aggregate to patch level.
    """
    home_i = np.diagonal(i, axis1=2, axis2=3)
    home_s = np.diagonal(s, axis1=2, axis2=3)
    commuter_s = s.sum(axis=3) - home_s
    commuter_i = i.sum(axis=3) - home_i
    return commuter_s, commuter_i, home_s, home_i


def simulate_batch(pop_sizes, od_matrix, realisations=10, extinctions=False, beta=2.5, psi=0.8, gamma=0.4,
                   perfect=False):
    """
    Run a batch of simulations, can set whether extinctions should be allowed or not.
    """
    t_max = 120
    t_delta = 1 / 12
    t_steps = int(t_max / t_delta)
    n = len(pop_sizes)
    s, i, r = (np.zeros(shape=(realisations, t_steps, n, n)),
               np.zeros(shape=(realisations, t_steps, n, n)),
               np.zeros(shape=(realisations, t_steps, n, n)))

    for k in range(realisations):
        sim = simulate(beta=beta, psi=psi, gamma=gamma, pop_sizes=pop_sizes, od_matrix=od_matrix, perfect=perfect)
        while sim[1].sum(axis=(0, 1, 2)) <= 100 and not extinctions:
            sim = simulate(beta=beta, psi=psi, gamma=gamma, pop_sizes=pop_sizes, od_matrix=od_matrix, perfect=perfect)
        s[k, ...], i[k, ...], r[k, ...] = sim[0], sim[1], sim[2]
    return s, i, r
