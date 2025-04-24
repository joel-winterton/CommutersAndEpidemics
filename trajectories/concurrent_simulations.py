import numpy as np
from epidemic_one_dim import Epidemic_1D
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from concurrent.futures import ProcessPoolExecutor

mpl.rcParams['figure.dpi'] = 300
cmap = mpl.colormaps['autumn']
patch_pop = 1000
number_of_patches = 50
t_max = 120
t_delta = 1 / 24

params = dict(n=patch_pop, t_max=t_max, k=number_of_patches, t_delta=t_delta)
p_c = 0.2

n_samples = 30
adventurer_props = np.array([0, 0.01, 0.02, 0.04, 0.05, 0.1, 0.125, 0.25, 0.35, 0.45, 0.5])
n_points = len(adventurer_props)


def run_simulation(point_and_sample):
    point, sample = point_and_sample
    p_a = adventurer_props[point]
    epidemic = Epidemic_1D(routine_type='simple_two', **params)
    epidemic.set_routine_params(dict(p_a=p_a, p_c=p_c))
    res = epidemic.simulate()
    peak_time = (res[1].sum(axis=-1)).argmax(axis=0) * t_delta
    return point, sample, peak_time


if __name__ == "__main__":
    alt_series = np.zeros((n_points, n_samples, number_of_patches))
    all_jobs = [(point, sample) for point in range(n_points) for sample in range(n_samples)]

    with ProcessPoolExecutor(max_workers=5) as executor:
        for point, sample, peak_time in executor.map(run_simulation, all_jobs):
            alt_series[point, sample, :] = peak_time

    print(alt_series)
    alt_series.tofile('alt_series.bin')
