import numpy as np
from copy import deepcopy
import scipy.optimize as optimize
import matplotlib.pyplot as plt

"""
Code for basic epidemic using standard meta-population SIR model and likelihood inference.
"""
rng = np.random.default_rng(seed=100)
def simulate(beta, gamma, psi, flow_matrix, population_sizes, seed_patch, t_max=100, t_delta=1):
    # state and initial conditions
    n = len(population_sizes)
    time_size = int(t_max // t_delta)
    s, i, r = (np.zeros(shape=(time_size, n), dtype=int),
               np.zeros(shape=(time_size, n), dtype=int),
               np.zeros(shape=(time_size, n), dtype=int))

    s[0, :] = population_sizes
    s[0, seed_patch] -= 1
    i[0, seed_patch] += 1
    # simulation
    foi = make_foi_fn(flow_matrix, population_sizes)
    for t in range(time_size - 1):
        delta_infections = rng.binomial(s[t, :], 1 - np.exp(-t_delta * foi(i[t, :], beta, psi)))
        delta_recoveries = rng.binomial(i[t, :], 1 - np.exp(-t_delta * gamma))

        s[t + 1, :] = s[t, :] - delta_infections
        i[t + 1, :] = i[t, :] + delta_infections - delta_recoveries
        r[t + 1, :] = r[t, :] + delta_recoveries
    return s, i, r


def make_foi_fn(flow_matrix, population_sizes, bulk=False):
    # so we can sum over entire matrix without having to account for j=i.
    np.fill_diagonal(flow_matrix, 0)

    def foi_fn(infections, beta, psi):
        between = (1 - psi) * np.dot(flow_matrix.T, infections / population_sizes)
        within = psi * infections
        return beta / population_sizes * (between + within)

    def bulk_foi_fn(infections, beta, psi):
        # Repeat data to allow bulk operations across time axis as well as patch axis
        t_max = infections.shape[0]
        population_sizes_bulk = np.repeat(population_sizes[None, ...], t_max, axis=0)
        flow_matrix_bulk = np.repeat(flow_matrix.T[None, ...], t_max, axis=0)

        between = (1 - psi) * np.einsum('ijk,ik->ij', flow_matrix_bulk, infections / population_sizes_bulk)
        within = psi * infections

        return beta / population_sizes_bulk * (within + between)

    if bulk:
        return bulk_foi_fn
    return foi_fn


def gamma_nll(gamma, state, t_delta=1):
    """"
    Negative log-likelihood function for gamma.
    """
    _, i, r = state
    deltas = np.diff(r, axis=0)
    ll_entries = np.log(1 - np.exp(-t_delta * gamma)) * deltas - t_delta * gamma * (i[:-1, :] - deltas)
    return -1 * ll_entries.sum(axis=(0, 1))


def transmission_nll(params, state, population_sizes, flow_matrix, t_delta=1):
    """
    Negative log-likelihood function for beta and psi.
    """
    beta, psi = params
    s, i, r = state
    # shifted_s = np.concatenate([s[1:, :], np.zeros(shape=(1, n))], axis=0)
    deltas = -1 * np.diff(s, axis=0)

    foi_fn = make_foi_fn(flow_matrix, population_sizes, bulk=True)
    foi = foi_fn(i, beta, psi)
    sane_foi = deepcopy(foi)
    sane_foi[foi == 0] = 0
    sane_foi[foi != 0] = np.exp(-t_delta * sane_foi[foi != 0])
    ll_entries = deltas * np.log(1 - sane_foi[:-1, :]) - t_delta * foi[:-1, :] * (s[:-1, :] - deltas)
    return -1 * ll_entries.sum(axis=(0, 1))


def fit_model(data, flow_matrix, population_sizes, t_delta=1):
    """
    Given tuple of (S,I,R) which each have shape (time, patch),
    estimate beta, gamma, psi using numerical optimisation of maximum likelihood.
    :param t_delta:
    :param data:
    :param flow_matrix:
    :param population_sizes:
    :return:
    """
    gamma = optimize.minimize(gamma_nll,
                              np.array([0.5]),
                              args=(data, t_delta),
                              method='L-BFGS-B',
                              bounds=[(0.001, 2)]).x[0]
    beta, psi = optimize.minimize(transmission_nll,
                                  np.array([1.8, 0.8]),
                                  args=(data, population_sizes, flow_matrix, t_delta),
                                  method='L-BFGS-B',
                                  bounds=[(0.01, 10), (0.001, 0.9)]).x
    return beta, gamma, psi


def visualise_fits(true_vals, estimated_vals, title=None):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(12, 4)
    if title:
        fig.suptitle(title)
    param_names = [r'$\beta$', r'$\gamma$', r'$\psi$']
    norm = plt.Normalize(true_vals[:, 0].min(), true_vals[:, 0].max())
    cmap = plt.get_cmap('rainbow')
    for index, param in enumerate(param_names):
        ax[index].scatter(true_vals[:, index], estimated_vals[:, index], s=4)
        ax[index].plot(true_vals[:, index], true_vals[:, index], c='g', alpha=0.2)
        ax[index].set_xlabel(f'True {param}')
        ax[index].set_ylabel(f'Estimated {param}')

    fig.tight_layout()
    plt.show()
