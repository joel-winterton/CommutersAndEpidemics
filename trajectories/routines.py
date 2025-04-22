"""
Routine generator functions.

"""
import numpy as np


def simple_routine(k, n, p_c, steps, boundaries):
    """
    Generates a home/work routine for each individual.
    """
    routine = np.zeros((steps, k, n), dtype=int)
    n_c = np.ceil(n * p_c).astype(int)
    home, work = boundaries
    # everyone at home
    routine[:home, :, :] = np.repeat(np.arange(k)[:, None], n, axis=1)
    # shift last n_c individuals 1 to the right during work hours, individuals at end of lattice stay home
    routine[home:, :, -n_c:] = np.repeat(np.minimum(np.arange(1, k + 1), k - 1)[:, None], n_c,
                                         axis=1)
    # everybody else stays home
    routine[home:, :, :n - n_c] = np.repeat(np.arange(k)[:, None], n - n_c, axis=1)
    return routine


def alternative_routine(k, n, p_c, p_a, steps, boundaries):
    """
    Generates a home/work routine plus a secondary route for n_a individuals in each site.
    """
    routine = simple_routine(k, n, p_c, steps, (boundaries[0], boundaries[1]))

    # sort edge case
    if p_a == 0:
        return routine

    # we can modify simple routine to avoid duplication
    n_a = np.ceil(n * p_c * p_a).astype(int)
    home, work, alternative = boundaries
    # n_a go to 3rd location
    routine[home + work:, :, -n_a:] = np.repeat(np.minimum(np.arange(2, k + 2), k - 1)[:, None], n_a, axis=1)
    routine[home:home + work, :, -n_a:] = np.repeat(np.minimum(np.arange(1, k + 1), k - 1)[:, None], n_a, axis=1)
    return routine


def simple_extensive_routine(k, n, n_c, steps, boundaries):
    """
    Generates a home/work routine for each individual.
    This routine sends individuals to n+1 and n+2 for work (50/50 split).
    """
    routine = simple_routine(k, n, n_c, steps, (boundaries[0], boundaries[1]))
    home, work = boundaries
    secondary_commuters = ''
