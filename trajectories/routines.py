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


def simple_two_routine(k, n, p_c, steps, boundaries, p_1, p_a):
    """
    Generates a home/work routine for each individual.
    This routine sends individuals to n+1 and n+2 for work (p_1, 1-p_1) split.
    """
    routine = np.zeros((steps, k, n), dtype=int)
    home, work = boundaries
    n_c = np.ceil(n * p_c).astype(int)

    c_1 = np.ceil(n * p_c * p_1).astype(int)
    c_2 = n_c - c_1
    a_1 = np.ceil(c_1 * p_a).astype(int)
    a_2 = np.ceil(c_2 * p_a).astype(int)
    # select commuters to go to 1 and 2 during work
    selected_mask = np.zeros(n_c, dtype=bool)
    selected_indices = np.random.choice(n_c, size=c_1, replace=False)
    selected_mask[selected_indices] = True
    commuter_1 = selected_indices + n - n_c
    commuter_2 = np.argwhere(~selected_mask).flatten() + n - n_c
    # everyone at home
    routine[:home, :, :] = np.repeat(np.arange(k)[:, None], n, axis=1)
    # people not working stay at home
    routine[home:, :, :n - n_c] = np.repeat(np.arange(k)[:, None], n - n_c, axis=1)
    # first set of commuters shifted 1 to the right
    routine[home: home + work, :, commuter_1] = np.repeat(np.minimum(np.arange(1, k + 1), k - 1)[:, None], c_1, axis=1)
    routine[home: home + work, :, commuter_2] = np.repeat(np.minimum(np.arange(2, k + 2), k - 1)[:, None], c_2, axis=1)

    if p_a == 0:
        return routine
    # now select adventurers from these and move them to other work location for last bit of day
    adventurers_1 = np.random.choice(commuter_1, size=a_1, replace=False)
    adventurers_2 = np.random.choice(commuter_2, size=a_2, replace=False)

    routine[home + work:, :, adventurers_1] = np.repeat(np.minimum(np.arange(2, k + 2), k - 1)[:, None], a_1, axis=1)
    routine[home + work:, :, adventurers_2] = np.repeat(np.minimum(np.arange(1, k + 1), k - 1)[:, None], a_2, axis=1)
    return routine