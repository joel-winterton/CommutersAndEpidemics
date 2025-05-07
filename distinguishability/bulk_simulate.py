"""
I've been running a lot of simulations where I am dependent on the sample size, so here I'm just running 50 simulations for a parameter
and then saving it to avoid waiting around a lot.
"""
from base_model import simulate as base_simulate
from model import simulate as two_way_simulate
from grab_data import *
from time import time as curr_time

def simulate(r0, incubation_period, t_max, sim='oneway', model='random', dataset='CENSUS_LAD11', samples=50,
             t_delta=1 / 12):
    simulator = base_simulate if sim == 'oneway' else two_way_simulate

    flow_matrix = get_matrix(dataset, full=True)
    population_sizes = get_population_sizes(dataset)
    params = dict(beta=r0 / incubation_period, gamma=1 / incubation_period, psi=2 / 3,
                  flow_matrix=flow_matrix,
                  population_sizes=population_sizes,
                  t_delta=t_delta,
                  t_max=t_max, seed=100, seed_amount=10)
    t_steps = np.floor(params['t_max'] / params['t_delta']).astype(int)
    i_series = np.zeros((samples, t_steps, len(population_sizes)))
    s_series = np.zeros((samples, t_steps, len(population_sizes)))
    for j in range(samples):
        print(f'Simulating {j} of {samples}')
        s, i, *_ = simulator(**params, model=model)
        while i.sum(axis=(0, 1, 2)) / (population_sizes.sum()) < 0.1:
            print(f"Extinction, restarting simulation {j} of {samples} ")
            print(i.sum(axis=(0, 1, 2)) / (population_sizes.sum()))
            s, i, *_ = simulator(**params, model=model)
        # shape in both arrays is (time, patch, subpatch)
        s_series[j] = s.sum(axis=-1)
        i_series[j] = i.sum(axis=-1)
    return s_series, i_series


# base simulate

incubation = 3
time = 200
t_delta = 1 / 12
samples = 50
for R0 in [1.3, 1.8, 2.0, 2.5, 3.0, 4.0]:
    print(f'Simulating R0 = {R0}')
    S_o, I_o = simulate(r0=R0, incubation_period=incubation,
                        t_max=time, t_delta=t_delta,
                        sim='oneway', model='perfect',
                        samples=samples)
    start = curr_time()
    print(f"Saving one-way perfect simulation, which has original shape of {S_o.shape}")
    np.save(
        f'simulation_data/S_model=perfect_oneway, r0={R0}, incubation={incubation}, max_time={time}, time_period = {1 / t_delta}, samples={samples}, shape={S_o.shape}',
        S_o)
    np.save(
        f'simulation_data/I_model=perfect_oneway, r0={R0}, incubation={incubation}, max_time={time}, time_period = {1 / t_delta}, samples={samples}, shape={S_o.shape}',
        I_o)
    S_t, I_t = simulate(r0=R0, incubation_period=incubation,
                        t_max=time, t_delta=t_delta,
                        sim='twoway', model='perfect',
                        samples=samples)

    print(f"Saving two-way perfect simulation, which has original shape of {S_t.shape}")
    np.save(
        f'simulation_data/S_model=perfect_twoway, r0={R0}, incubation={incubation}, max_time={time}, time_period = {1 / t_delta}, samples={samples}, shape={S_o.shape}',
        S_t)
    np.save(
        f'simulation_data/I_model=perfect_twoway, r0={R0}, incubation={incubation}, max_time={time}, time_period = {1 / t_delta}, samples={samples}, shape={S_o.shape}',
        I_t)
    end = curr_time()
    print(f"Process took {end - start:.2f} seconds")