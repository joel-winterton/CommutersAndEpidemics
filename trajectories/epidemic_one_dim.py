"""
Toy 1-dimensional epidemic, to start playing around with trajectories vs OD matrices.

"""
import numpy as np
from routines import simple_routine, alternative_routine, simple_two_routine


class Epidemic_1D:
    """
    Wrapper for parameters to combine routine with simulation, for consistency.
    """

    def __init__(self, routine_type, n=500, k=20, t_max=120, t_delta=1 / 24, beta=1.2, gamma=0.2, extinction=False):
        """
        routine_type: 'simple', 'alternative' see routines.py for descriptions of these routines.
        n: Initial population size for each site (homogenous).
        k: Number of sites.
        t_max: Maximum simulation time.
        t_delta: Simulation time step.
        """
        self.n, self.k, self.t_max, self.t_delta = n, k, t_max, t_delta
        self.beta, self.gamma = beta, gamma
        self.routine_type = routine_type
        self.extinction = extinction

        self.time_steps = np.floor(self.t_max / self.t_delta).astype(int)
        self.day_steps = int(1 / t_delta)
        self.params = self.get_routine_default_params()

    def get_routine(self):
        """
        Generates routine from set parameters.
        Function will be called at start of simulation so no need to update it every time a parameter is updated.
        """
        if self.routine_type == 'simple':
            return simple_routine(**self.params)
        elif self.routine_type == 'alternative':
            return alternative_routine(**self.params)
        elif self.routine_type == 'simple_two':
            return simple_two_routine(**self.params)

    def get_routine_default_params(self):
        """
        Store for default parameters for each routine.
        """
        if self.routine_type == 'simple':
            return dict(k=self.k, n=self.n, p_c=0.2, boundaries=(16, 8), steps=self.day_steps)
        elif self.routine_type == 'alternative':
            return dict(k=self.k, n=self.n, p_c=0.2, p_a=0.2, boundaries=(14, 8, 2), steps=self.day_steps)
        elif self.routine_type == 'simple_two':
            return dict(k=self.k, n=self.n, p_c=0.2, boundaries=(16, 8), steps=self.day_steps, p_1=0.5, p_a=0)

    def set_routine_params(self, params):
        """
        Function to change routine parameters.
        """
        for k, v in params.items():
            if k not in self.params:
                raise ValueError(f"Unknown routine parameter '{k}'")

            self.params[k] = v

    def simulate(self):
        routine = self.get_routine()
        # creates a new lattice each simulation, since lattice state updated through time
        lattice = Lattice_1D(routine=routine, t_max=self.t_max,
                             t_delta=self.t_delta,
                             beta=self.beta, gamma=self.gamma)
        return lattice.simulate()


class Lattice_1D:
    def __init__(self, t_max, t_delta, routine, beta, gamma):
        """
        Lattice proportions are determined from passed routine array.
        p_c: Proportion of individuals who commute.
        n_a: Proportion of commuters who then go to 3rd location.
        beta: Yeah.
        gamma: Yeah.
        """
        self.beta, self.gamma = beta, gamma
        self.t_delta, self.t_max = t_delta, t_max
        self.routine = routine

        self.time_steps, self.k, self.n = routine.shape
        self.day_steps = int(1 / self.t_delta)
        self.total_pop = self.k * self.n
        self.time_steps = t_max * self.day_steps
        if self.time_steps != t_max * self.day_steps:
            raise ValueError("Given routine times are not consistent with passed time parameters!")
        # Indicator matrices (patch number, individual number)
        self.s, self.i, self.r = (np.full(shape=(self.time_steps, self.k, self.n), fill_value=1),
                                  np.zeros(shape=(self.time_steps, self.k, self.n)),
                                  np.zeros(shape=(self.time_steps, self.k, self.n)))
        # seed
        self.s[0, 0, 0] = 0
        self.i[0, 0, 0] = 1

    def simulate(self):
        """
        Run epidemic simulation.
        """
        patch_list = np.arange(self.k)[:, None, None]
        t = 0
        time_step = 0
        while time_step < self.time_steps - 1:
            hour = time_step % self.day_steps
            masks = self.routine[hour][None, :, :] == patch_list
            rng_vals_a = np.random.rand(*masks.shape)
            rng_vals_b = np.random.rand(*masks.shape)
            infecteds = np.logical_and(masks, self.i[time_step]).sum(axis=(1, 2))
            if infecteds.sum() == 0 and time_step < 0.25 * self.time_steps:
                print('Extinction event occurred, restarting')
                return self.simulate()

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
