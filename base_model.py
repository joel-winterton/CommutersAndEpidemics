"""
Base distinguishable models.
"""
import numpy as np
from copy import deepcopy

rng = np.random.default_rng()


def make_foi(beta, psi, flow_matrix, population_sizes, model):
    """
    Force of infections based off subpatch model.
    """

    def random_foi(infections):
        total_infections = infections.sum(axis=1)
        within = psi * total_infections
        between = (1 - psi) * np.dot(flow_matrix.T, total_infections / population_sizes)
        patch_foi = beta / population_sizes * (between + within)
        return np.repeat(patch_foi[..., np.newaxis], flow_matrix.shape[0], axis=1)

    def perfect_foi(infections):
        within = psi * infections.sum(axis=1)
        between = (1 - psi) * infections.sum(axis=0)
        patch_foi = beta / population_sizes * (between + within)
        return np.repeat(patch_foi[..., np.newaxis], flow_matrix.shape[0], axis=1)

    if model == 'random':
        return random_foi
    elif model == 'perfect':
        return perfect_foi


def make_state(flow_matrix, population_sizes, seed, seed_amount, time_steps):
    x_ini, y_ini, z_ini = deepcopy(flow_matrix), np.zeros(shape=flow_matrix.shape), np.zeros(shape=flow_matrix.shape)

    x_ini[seed, seed] -= seed_amount
    y_ini[seed, seed] += seed_amount

    x, y, z = (np.zeros(shape=(time_steps, *flow_matrix.shape), dtype=int),
               np.zeros(shape=(time_steps, *flow_matrix.shape), dtype=int),
               np.zeros(shape=(time_steps, *flow_matrix.shape), dtype=int))

    x[0, ...] = x_ini
    y[0, ...] = y_ini
    z[0, ...] = z_ini
    return x, y, z


def simulate(beta, psi, gamma, seed, seed_amount, t_max, t_delta, flow_matrix, population_sizes, model):
    t_steps = int(t_max / t_delta)
    x, y, z = make_state(flow_matrix, population_sizes, seed, seed_amount, t_steps)
    foi_fn = make_foi(beta, psi, flow_matrix, population_sizes, model)
    for t in range(t_steps - 1):
        curr_x, curr_y, curr_z = x[t, ...], y[t, ...], z[t, ...]
        foi = foi_fn(curr_y)
        delta_infection = rng.binomial(curr_x, 1 - np.exp(-t_delta * foi))
        delta_recovery = rng.binomial(curr_y, 1 - np.exp(-t_delta * gamma))
        x[t + 1, ...] = curr_x - delta_infection
        y[t + 1, ...] = curr_y + delta_infection - delta_recovery
        z[t + 1, ...] = curr_z + delta_recovery
    return x, y, z
