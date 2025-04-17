"""
Toy 1-dimensional epidemic, to start playing around with trajectories vs OD matrices.

"""
import numpy as np


def simple_routine(k, n, n_c, steps, boundaries):
    """
    Generates a home/work routine for each individual.
    """
    routine = np.zeros((steps, k, n), dtype=int)
    home, work = boundaries
    # everyone at home
    routine[:home, :, :] = np.repeat(np.arange(k)[:, None], n, axis=1)
    # shift last n_c individuals 1 to the right during work hours, individuals at end of lattice stay home
    routine[home:, :, -n_c:] = np.repeat(np.minimum(np.arange(1, k + 1), k - 1)[:, None], n_c,
                                         axis=1)
    # everybody else stays home
    routine[home:, :, :n - n_c] = np.repeat(np.arange(k)[:, None], n - n_c, axis=1)
    return routine


def alternative_routine(k, n, n_c, n_a, steps, boundaries):
    """
    Generates a home/work routine plus a secondary route for n_a individuals in each site.
    """
    # we can modify simple routine to avoid duplication
    routine = simple_routine(k, n, n_c, steps, (boundaries[0], boundaries[1]))
    home, work, alternative = boundaries
    # n_c - n_a go home
    routine[steps - home - work:, :, n - n_c:n - n_a] = np.repeat(np.arange(k)[:, None], n_c - n_a, axis=1)
    # n_a go to 3rd location
    routine[steps - home:, :, -n_a:] = np.repeat(np.minimum(np.arange(2, k + 2), k - 1)[:, None], n_a, axis=1)
    return routine


class Lattice_1D:
    def __init__(self, k=10, n=100, n_c=20, n_a=5, beta=2.4, gamma=0.2, t_max=100, t_delta=1 / 24,
                 simple_boundaries=(16, 8),
                 alternative_boundaries=(14, 8, 2)):
        """
        k: Number of lattice sites.
        n: Site population sizes.
        n_c: Number of commuters in each site.
        n_a: Number of commuters who particpate in the alternative trajectory.
        beta: Yeah.
        gamma: Yeah.
        """
        self.beta, self.gamma = beta, gamma
        self.k = k
        self.n = n
        self.n_c = n_c
        self.n_a = n_a
        self.t_delta = t_delta
        self.t_max = t_max
        self.simple_boundaries = simple_boundaries
        self.alternative_boundaries = alternative_boundaries

        self.total_pop = k * n
        self.time_steps = np.floor(t_max // t_delta).astype(int)
        # Indicator matrices (patch number, individual number)
        self.s, self.i, self.r = (np.full(shape=(self.time_steps, k, self.n), fill_value=1),
                                  np.zeros(shape=(self.time_steps, k, self.n)),
                                  np.zeros(shape=(self.time_steps, k, self.n)))
        # seed
        self.s[0, 0, [0, 1, 2]] = 0
        self.i[0, 0, [0, 1, 2]] = 1

        pass

    def get_routine(self, routine, day_steps):
        """
        Return the relevant routine for simulation.
        """
        if routine == 'simple':
            return simple_routine(self.k, self.n, self.n_c, day_steps, self.simple_boundaries)
        elif routine == 'alternative':
            return alternative_routine(self.k, self.n, self.n_c, self.n_a, day_steps, self.alternative_boundaries)

    def simulate(self, routine='simple'):
        """
        Run epidemic on a given routine.
        """
        day_steps = int(1 / self.t_delta)
        patch_list = np.arange(self.k)[:, None, None]
        routine = self.get_routine(routine=routine, day_steps=day_steps)
        t = 0
        time_step = 0
        while time_step < self.time_steps - 1:
            hour = time_step % day_steps
            masks = routine[hour][None, :, :] == patch_list
            rng_vals_a = np.random.rand(*masks.shape)
            rng_vals_b = np.random.rand(*masks.shape)
            infecteds = np.logical_and(masks, self.i[time_step]).sum(axis=(1, 2))
            infection_probabilities = (1 - np.exp(-self.t_delta * self.beta * infecteds / self.n))[:, None, None]
            new_infection_mask = np.any(
                masks & (rng_vals_a < infection_probabilities) & self.s[time_step][None, :, :].astype(
                    bool), axis=0)
            recovery_probability = 1 - np.exp(-self.t_delta * self.gamma)
            new_recoveries_mask = np.any(
                masks & (rng_vals_b < recovery_probability) & self.i[time_step][None, :, :].astype(bool), axis=0)

            self.s[time_step + 1] = self.s[time_step]
            self.i[time_step + 1] = self.i[time_step]
            self.r[time_step + 1] = self.r[time_step]
            self.s[time_step + 1, new_infection_mask] = 0
            self.i[time_step + 1, new_infection_mask] = 1
            self.i[time_step + 1, new_recoveries_mask] = 0
            self.r[time_step + 1, new_recoveries_mask] = 1
            t += self.t_delta
            time_step += 1
        return self.s, self.i, self.r, np.arange(0, self.t_max, self.t_delta)
