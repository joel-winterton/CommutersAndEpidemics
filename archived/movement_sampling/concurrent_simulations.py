"""
I want to compare epidemics across a parameter range, and the movement sampler model takes a bit of time,
so this script parallelizes this across my parameter range to save some time.
"""
import concurrent.futures
import csv

from movement_samplers import RandomCommuterSampler, PerfectCommuterSampler
from movement_epidemic import MovementEpidemic
from basic_epidemic import fit_model
from grab_data import get_matrix, get_population_sizes
from itertools import product
import numpy as np

n = (5, 5, 2)
SAMPLER = PerfectCommuterSampler
DATASET = 'CENSUS_SUBSAMPLED'

# generate list of all combinations of parameters I want to try
parameters = np.array(list(product(np.linspace(1, 2, n[0]),
                                   np.linspace(0.2, 0.8, n[1]),
                                   np.linspace(0.5, 0.9, n[2]))))
# cover a bit more interesting cases quicker
np.random.shuffle(parameters)

od_matrix = get_matrix(DATASET)
pop_sizes = get_population_sizes(DATASET)


def tidy_simulate_and_fit(params):
    b, g, p = params
    epi = MovementEpidemic(RandomCommuterSampler,
                           od_matrix,
                           pop_sizes,
                           beta=b,
                           gamma=g,
                           psi=p,
                           t_max=100)
    sim = epi.simulate()
    s,i,r,infected_commuters = sim
    np.savetxt(f'PERFECT_B={b}_G')


def execute():
    i = 0
    for param in parameters:
        print(f'Done {i}')
        result = tidy_simulate_and_fit(param)
        with open("perfect_commuter_results.csv", mode="a", newline="") as file:  # 'a' mode for appending
            writer = csv.writer(file)
            writer.writerows([[*param, *result]])
        i += 1


execute()
