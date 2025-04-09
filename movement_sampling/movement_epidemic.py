from movement_sampling.movement_samplers import Sampler
import numpy as np

rng = np.random.default_rng(seed=100)


class MovementEpidemic:
    def __init__(self, sampler, od_matrix: type(Sampler), population_sizes, t_max, beta, psi, gamma, seed_patch=0):
        self.population_sizes = population_sizes
        self.movement_sampler = sampler(od_matrix, population_sizes)
        self.od_matrix = od_matrix
        self.number_of_patches = od_matrix.shape[0]
        self.infected_commuters = np.zeros(shape=(t_max, self.number_of_patches, self.number_of_patches), dtype=float)
        self.t_max = t_max
        self.beta = beta
        self.psi = psi
        self.gamma = gamma

        # index: (patch, time, individual)
        self.s = [np.full(shape=(t_max, self.population_sizes[i]), fill_value=1) for i in range(self.number_of_patches)]
        self.i = [np.zeros(shape=(t_max, self.population_sizes[i])) for i in range(self.number_of_patches)]
        self.r = [np.zeros(shape=(t_max, self.population_sizes[i])) for i in range(self.number_of_patches)]

        self.seed(seed_patch)

    def seed(self, patch_index):
        """
        Infect a single random individual in given patch at t=0.
        :param patch_index:
        :return:
        """
        individual = rng.integers(0, self.population_sizes[patch_index])
        self.s[patch_index][0, individual] = 0
        self.i[patch_index][0, individual] = 1

    def foi_fn(self, t, state) -> np.ndarray:
        """
        Return vector of FOIs at time t, e.g. to be applied at timestep after t.
        :param state:
        :param t:
        :return:
        """
        (susceptibles, infecteds), (s_indices, i_indices), exerted_foi = state

        between = (1 - self.psi) * exerted_foi
        within = self.psi * infecteds
        return self.beta / self.population_sizes * (between + within)

    def update_states(self, infections, recoveries, s_indices, i_indices, t):
        """
        Given new infections and recoveries, update our state.
        :param infections:
        :param recoveries:
        :param s_indices:
        :param i_indices:
        :param t:
        :return:
        """
        for j in range(self.number_of_patches):
            self.s[j][t + 1, :] = self.s[j][t, :]
            self.i[j][t + 1, :] = self.i[j][t, :]
            self.r[j][t + 1, :] = self.r[j][t, :]

            new_infections = rng.choice(s_indices[j], infections[j], replace=False)
            new_recoveries = rng.choice(i_indices[j], recoveries[j], replace=False)
            # Update individual states
            self.s[j][t + 1, new_infections] = 0
            self.i[j][t + 1, new_infections] = 1
            self.i[j][t + 1, new_recoveries] = 0
            self.r[j][t + 1, new_recoveries] = 1

    def get_current_state(self, t):
        """
        Get current state of epidemic.
        :param t:
        :return:
        """
        susceptibles = np.zeros(shape=self.number_of_patches, dtype=int)
        infecteds = np.zeros(shape=self.number_of_patches, dtype=int)
        s_indices, i_indices = [], []
        external_foi = np.zeros(shape=(self.number_of_patches, self.number_of_patches), dtype=float)
        for k in range(self.number_of_patches):
            infecteds[k] = self.i[k][t, :].sum()
            susceptibles[k] = self.s[k][t, :].sum()
            s_indices.append(np.argwhere(self.s[k][t, :] == 1).flatten())
            i_indices.append(np.argwhere(self.i[k][t, :] == 1).flatten())
            infection_state = self.i[k][t, :]
            movement = self.movement_sampler.sample(k)
            external_foi[k, :] = np.dot(movement, infection_state)

        self.infected_commuters[t, ...] = external_foi
        exerted_foi = external_foi.sum(axis=0) - external_foi.diagonal()

        return (susceptibles, infecteds), (s_indices, i_indices), exerted_foi

    def step(self, t):
        """
        Run a single time step.
        """
        if t >= self.t_max:
            return False
        state = self.get_current_state(t)
        (susceptibles, infecteds), (s_indices, i_indices), exerted_foi = state
        if t < 0.1 * self.t_max:
            if infecteds.sum() == 0:
                return False
        foi = self.foi_fn(t, state)
        probabilities = 1 - np.exp(-foi)
        infection_deltas = rng.binomial(susceptibles, probabilities)
        recovery_deltas = rng.binomial(infecteds, 1 - np.exp(-self.gamma))
        self.update_states(infection_deltas, recovery_deltas, s_indices, i_indices, t)
        return True

    def aggregate_states(self):
        """
        Aggregate individual states to give total SIR timeseries for each patch.
        :return:
        """
        s, i, r = (np.zeros(shape=(self.t_max, self.number_of_patches), dtype=int),
                   np.zeros(shape=(self.t_max, self.number_of_patches), dtype=int),
                   np.zeros(shape=(self.t_max, self.number_of_patches), dtype=int))

        for t in range(self.t_max):
            for patch in range(self.number_of_patches):
                s[t, patch] = self.s[patch][t, :].sum()
                i[t, patch] = self.i[patch][t, :].sum()
                r[t, patch] = self.r[patch][t, :].sum()

        return s, i, r, self.infected_commuters

    def simulate(self):
        """
        Populate time states.
        :return:
        """
        t = 0
        while t < self.t_max - 1:
            print(f"Currently at step {t}", end='\r')
            step = self.step(t)
            # if extinction happens, restart simulation so we can automate over epidemics
            if step:
                t += 1
            else:
                t = 0

        print("Simulation finished, aggregating states.")
        return self.aggregate_states()
