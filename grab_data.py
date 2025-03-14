"""
This exposes a function that will give a numpy commuter matrix to avoid doing this many times in each analysis.
"""
import pandas as pd
import numpy as np

mock_commuter_flow = np.array([
    [100, 10, 2],
    [15, 150, 10],
    [35, 5, 200]
], dtype=int)

mock_non_commuter_counts = np.array([25, 10, 40])


def get_matrix(dataset='LAD11'):
    """"
    Returns commuter matrix as numpy ndarray. Has 2011 LAD census and mock data.
    Read more about 2011 data: `2011_census_data/README.md`
    """
    if dataset == 'LAD11':
        data = pd.read_csv('2011_census_data/WF02EW LAD2011.csv', index_col=0, skiprows=9)
        data.drop(data.index[range(346, len(data.index))], inplace=True)
        return data.values.astype(float)
    if dataset == 'MOCK':
        return mock_commuter_flow


def get_population_sizes(dataset='LAD11'):
    """"
    Returns population size numpy array. Has 2011 LAD census and mock data.
    Read more about 2011 data: `2011_census_data/README.md`
    """
    if dataset == 'LAD11':
        data = pd.read_csv('2011_census_data/QS102EW LAD2011.csv', index_col=0, skiprows=8)
        data.drop(data.index[range(346, len(data.index))], inplace=True)
        # drop footer text
        return data.values.flatten().astype(float)
    if dataset == 'MOCK':
        return mock_commuter_flow.sum(axis=1) + mock_non_commuter_counts
