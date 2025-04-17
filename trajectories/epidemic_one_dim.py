"""
Toy 1-dimensional epidemic, to start playing around with trajectories vs OD matrices.

"""
import numpy as np


class Lattice_1D:
    def __init__(self, k=10, n=100, n_c=20, beta=2.4, gamma=0.2, t_max=100, t_delta=1 / 24,
                 simple_boundaries=(16, 8),
                 alternative_boundaries=(14, 8, 2)):
        """
        k: Number of lattice sites.
        n: Site population sizes.
        n_c: Number of commuters in each site.
        beta: Yeah.
        gamma: Yeah.
        """
        self.beta, self.gamma = beta, gamma
        self.k = k
        self.n = n
        self.n_c = n_c
        self.t_delta = t_delta
        self.t_max = t_max
        self.simple_boundaries = simple_boundaries
        self.alternative_boundaries = alternative_boundaries

        self.total_pop = k * n
        self.time_steps = np.round(t_max // t_delta, 0).astype(int)
        # Indicator matrices (patch number, individual number)
        self.s, self.i, self.r = (np.full(shape=(self.time_steps, k, self.n), fill_value=1),
                                  np.zeros(shape=(self.time_steps, k, self.n)),
                                  np.zeros(shape=(self.time_steps, k, self.n)))
        # seed
        self.s[0, 0, [0, 1, 2]] = 0
        self.i[0, 0, [0, 1, 2]] = 1

        pass

    def generate_trajectories(self):
        day_steps = int(1 / self.t_delta)
        # Simple trajectory
        simple_routine = np.zeros((day_steps, self.k, self.n), dtype=int)
        home, work = self.simple_boundaries
        simple_routine[:home, :, :] = np.repeat(np.arange(self.k)[:, None], self.n, axis=1)
        # shift first n_c individuals 1 to the right during work hours
        simple_routine[home:, :, :self.n_c] = np.repeat(np.minimum((np.arange(self.k) + 1), self.k)[:, None], self.n_c,
                                                        axis=1)
        simple_routine[home:, :, self.n_c:] = np.repeat(np.arange(self.k)[:, None], self.n - self.n_c, axis=1)

        # Alternative trajectory
        alternative_routine = np.zeros((day_steps, self.k, self.n), dtype=int)
        home, work, alternative = self.alternative_boundaries
        alternative_routine[:home, :, :] = np.repeat(np.arange(self.k)[:, None], self.n, axis=1)
        alternative_routine[home:home + work, :, :self.n_c] = np.repeat(
            np.minimum((np.arange(self.k) + 1), self.k)[:, None],
            self.n_c,
            axis=1)
        alternative_routine[home:home + work, :, self.n_c:] = np.repeat(np.arange(self.k)[:, None], self.n - self.n_c,
                                                                        axis=1)

        alternative_routine[home + work:, :, :self.n_c] = np.repeat(
            np.minimum((np.arange(self.k) + 2), self.k)[:, None],
            self.n_c,
            axis=1)
        alternative_routine[home + work:, :, self.n_c:] = np.repeat(np.arange(self.k)[:, None], self.n - self.n_c,
                                                                    axis=1)

        return simple_routine, alternative_routine

    def simulate(self, route='simple'):
        day_steps = int(1 / self.t_delta)
        patch_list = np.arange(self.k)[:, None, None]
        simple_routine, alternative_routine = self.generate_trajectories()
        routine = simple_routine if route == 'simple' else alternative_routine
        t = 0
        time_step = 0
        while t < self.t_max - self.t_delta:
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
        return self.s, self.i, self.r
