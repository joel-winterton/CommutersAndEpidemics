import numpy as np
from copy import deepcopy

from scipy.linalg import eigh


def make_perfect_foi(beta, psi, od_matrix, pop_sizes, tau_0, tau_1, delta_t):
    """
    Make force of infection function for perfect commuters model.
    """

    def foi_fn(time_index: int, state):
        t = delta_t * time_index
        _, y, _ = state
        n = y.shape[0]
        w = tau_0 < t % 1 <= tau_1
        if w:
            work = beta / pop_sizes * y.sum(axis=0)
            foi = np.repeat(work[np.newaxis, ...], n, axis=0)
        else:
            home = beta / pop_sizes * y.sum(axis=1)
            foi = np.repeat(home[..., np.newaxis], n, axis=1)
        return foi

    return foi_fn


def make_random_foi(beta, psi, od_matrix, pop_sizes, tau_0, tau_1, delta_t):
    """
    Make force of infection function for random commuters model.
    TODO: rewrite this model since psi definition has been changed.
    """
    non_commuter_counts = pop_sizes - od_matrix.sum(axis=1)

    def foi_fn(time_index: int, state):
        t = delta_t * time_index
        _, y, _ = state
        n = y.shape[0]
        w = tau_0 < t % 1 <= tau_1
        y_tot = y.sum(axis=1)
        if w:
            home = y_tot * non_commuter_counts / pop_sizes
            work = (np.dot(od_matrix.T, y_tot / pop_sizes))
            rates = beta / pop_sizes * (home + work)
            foi = np.repeat(rates[..., np.newaxis], n, axis=1)
        else:
            home = beta * y_tot / pop_sizes
            foi = np.repeat(home[..., np.newaxis], n, axis=1)
        return foi

    return foi_fn


def make_one_way_perfect_foi(beta, psi, od_matrix, pop_sizes, tau_0, tau_1, delta_t):
    """
    This should take one_way params.
    """
    np.fill_diagonal(od_matrix, 0)

    def foi_fn(time_index: int, state):
        _, y, _ = state
        within = y.sum(axis=1)
        between = y.sum(axis=0) - np.diagonal(y)
        foi = beta / pop_sizes * (psi * within + (1 - psi) * between)
        return np.repeat(foi[..., np.newaxis], y.shape[0], axis=1)

    return foi_fn


def make_one_way_random_foi(beta, psi, od_matrix, pop_sizes, tau_0, tau_1, delta_t):
    """
    This should take one_way params.
    """
    np.fill_diagonal(od_matrix, 0)

    def foi_fn(time_index: int, state):
        _, y, _ = state
        y_tot = y.sum(axis=1)
        between = (1 - psi) * np.dot(od_matrix.T, y_tot / pop_sizes)
        within = psi * y_tot
        foi = beta / pop_sizes * (between + within)
        return np.repeat(foi[..., np.newaxis], y.shape[0], axis=1)

    return foi_fn


def make_state(pop_sizes, od_matrix, seed, seed_amount, time_steps):
    # TODO make timestep and t_max automatically compatible.
    # TODO automate t_delta calculation based off taus.
    x_ini = deepcopy(od_matrix)
    np.fill_diagonal(x_ini, 0)
    commuters = x_ini.sum(axis=1)
    non_commuters = pop_sizes - commuters
    np.fill_diagonal(x_ini, non_commuters)
    y_ini, z_ini = np.zeros_like(od_matrix), np.zeros_like(od_matrix)

    x_ini[seed, seed] -= seed_amount
    y_ini[seed, seed] += seed_amount

    x, y, z = (np.zeros(shape=(time_steps, *od_matrix.shape), dtype=int),
               np.zeros(shape=(time_steps, *od_matrix.shape), dtype=int),
               np.zeros(shape=(time_steps, *od_matrix.shape), dtype=int))

    x[0, :] = x_ini
    y[0, :] = y_ini
    z[0, :] = z_ini
    return x, y, z


def simulate(beta, gamma, pop_sizes, od_matrix, psi=1, delta=1 / 12, t_max=120, seed=0, seed_amount=1, tau_0=9 / 24,
             tau_1=17 / 24, model='random'):
    time_steps = int(t_max / delta)
    x, y, z = make_state(pop_sizes, od_matrix, seed, seed_amount, time_steps)
    fois = {'random': make_random_foi,
            'perfect': make_perfect_foi,
            'random_oneway': make_one_way_random_foi,
            'perfect_oneway': make_one_way_perfect_foi, }
    foi_fn = fois[model](beta, psi, od_matrix, pop_sizes, tau_0, tau_1, delta)
    rng = np.random.default_rng()
    for time_step in range(time_steps - 1):
        foi = foi_fn(time_step, (x[time_step, ...], y[time_step, ...], z[time_step, ...]))
        delta_infection = rng.binomial(x[time_step, ...], 1 - np.exp(-delta * foi))
        # print(delta_infection)
        delta_recovery = rng.binomial(y[time_step, ...], 1 - np.exp(- delta * gamma))
        x[time_step + 1, ...] = x[time_step] - delta_infection
        y[time_step + 1, ...] = y[time_step] + delta_infection - delta_recovery
        z[time_step + 1, ...] = z[time_step] + delta_recovery
    t = np.linspace(0, t_max, time_steps)
    return x, y, z, t


def two_way_to_one_way(beta, gamma, tau_0, tau_1, od_matrix, pop_sizes):
    """
    Converts parameters used in 2 way model to 1 way parameters that have the same R0.
    """
    # create commuter matrices with (1) zero diagonal and (2) non-zero diagonals that sum rows to pop size.
    nonzero = deepcopy(od_matrix)
    zero = deepcopy(od_matrix)
    np.fill_diagonal(zero, 0)
    np.fill_diagonal(nonzero, 0)
    commuters = nonzero.sum(axis=1)
    non_commuters = pop_sizes - commuters
    np.fill_diagonal(nonzero, non_commuters)

    c_hat = eigh(np.dot(zero.T, np.diag(1 / pop_sizes)), eigvals_only=True)[-1]
    c_tilde = eigh(np.dot(nonzero.T, np.diag(1 / pop_sizes)), eigvals_only=True)[-1]

    psi = tau_1 - tau_0
    conversion = (psi + (1 - psi) * c_tilde) / (psi + (1 - psi) * c_hat)

    return dict(beta=beta * conversion, gamma=gamma, psi=psi)


def one_way_to_two_way(beta, gamma, psi, od_matrix, pop_sizes):
    """
    Converts parameters used in 2 way model to 1 way parameters that have the same R0.
    """
    # create commuter matrices with (1) zero diagonal and (2) non-zero diagonals that sum rows to pop size.
    nonzero = deepcopy(od_matrix)
    zero = deepcopy(od_matrix)
    np.fill_diagonal(zero, 0)
    np.fill_diagonal(nonzero, 0)
    commuters = nonzero.sum(axis=1)
    non_commuters = pop_sizes - commuters
    np.fill_diagonal(nonzero, non_commuters)

    c_hat = eigh(np.dot(zero.T, np.diag(1 / pop_sizes)), eigvals_only=True)[-1]
    c_tilde = eigh(np.dot(nonzero.T, np.diag(1 / pop_sizes)), eigvals_only=True)[-1]

    tau_0 = 9 / 24
    tau_1 = psi + tau_0
    conversion = (psi + (1 - psi) * c_hat) / (psi + (1 - psi) * c_tilde)

    return dict(beta=beta * conversion, gamma=gamma, tau_0=tau_0, tau_1=tau_1)


def two_way_r0(beta, gamma, tau_0, tau_1, od_matrix, pop_sizes):
    """
    Calculates R0 for two-way model
    """
    nonzero = deepcopy(od_matrix)
    np.fill_diagonal(nonzero, 0)
    commuters = nonzero.sum(axis=1)
    non_commuters = pop_sizes - commuters
    np.fill_diagonal(nonzero, non_commuters)

    psi = tau_1 - tau_0
    c_tilde = eigh(np.dot(nonzero.T, np.diag(1 / pop_sizes)), eigvals_only=True)[-1]
    return beta / gamma