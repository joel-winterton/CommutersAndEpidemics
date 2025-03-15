import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt, cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from numpy import genfromtxt


def tabulate(infection_matrix, patch_ids, shapefile):
    """
    Merge infection time series matrix and geopandas file into one dataframe that can be used
    for animation.
    :param shapefile:
    :param infection_matrix:
    :param patch_ids:
    :return:
    """
    time_series = infection_matrix.T
    dataframes = []

    for j in range(time_series.shape[0]):
        dataframes.append(pd.DataFrame({'t': np.arange(time_series.shape[1]), 'i': time_series[j, :],
                                        'LAD21CD': np.repeat(patch_ids[j], time_series.shape[1])}))
    infection_data = pd.concat(dataframes).merge(shapefile, on='LAD21CD', how='inner')
    infection_geo = gpd.GeoDataFrame(infection_data, geometry=infection_data.geometry)
    return infection_geo


def animate_simulation(time_series, patch_ids, shapefile, title=None, write=True):
    """
    Animates simulated disease spread.
    :param title:
    :param time_series:
    :param patch_ids:
    :param shapefile:
    :param write:
    :return:
    """
    df = tabulate(time_series, patch_ids, shapefile)
    t_max = df.shape[0]
    # Create the colormap using the min/max values
    vmin = df['i'].min()
    vmax = df['i'].max()
    COLORMAP = 'RdYlGn_r'
    cmap = cm.ScalarMappable(Normalize(vmin, vmax), COLORMAP)
    fig, [ax, cax] = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw={"width_ratios": [50, 1]})
    ax.set_axis_off()
    # Create the colorbar with colormap
    plt.colorbar(mappable=cmap, cax=cax, )
    cax.set_title('Prevalence')

    def plot_day(t: int):
        ax.clear()
        ax.set_axis_off()
        data = df[df['t'] == t]
        if title:
            ax.set_title(f"{title} t={t}")
        else:
            ax.set_title(f"t={t}")
        data.plot(column='i', ax=ax, cmap=COLORMAP)
        plt.close()
        return []

    animation = FuncAnimation(fig, plot_day, frames=t_max, repeat=False, interval=100)

    if write:
        print("Saving!")
        animation.save(f"{title}.gif", writer='imagemagick')
    else:
        plt.show()


PATH_TO_SIM = 'big_sims/infections_tmax=100_od=ons_sampler=perfectcommuter.csv'
data = genfromtxt(PATH_TO_SIM, delimiter=',')
infections = data[..., 1:]
lads = gpd.read_file("2021_commuter_data/shapefiles/LAD_DEC_2021_GB_BFC.shp")
commuter_patches = pd.read_csv("2021_commuter_data/commuter_matrix.csv", index_col=0).index.values
df = tabulate(infection_matrix=infections, patch_ids=commuter_patches, shapefile=lads)
animate_simulation(time_series=infections, shapefile=lads, patch_ids=commuter_patches, title=r'Perfect Commuter Epidemic $\beta = 1.8, \gamma = 0.3$ ')
