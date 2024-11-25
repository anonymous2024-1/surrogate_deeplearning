import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import string

from sklearn.cluster import KMeans
import networkx as nx
import scipy
from scipy.spatial.distance import cdist
from mpl_toolkits import basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata


def ecdf(x):
    """Calculate empirical cummulative density function
    Parameters
    ----------
    x : np.ndarray
        Array containing the data
    Returns
    -------
    x : np.ndarray
        Array containing the sorted metric values
    y : np.ndarray]
        Array containing the sorted cdf values
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def flatData(x, sortOpt=0):
    # sortOpt: 0: small to large, 1: large to small, -1: no sort
    xArrayTemp = x.flatten()
    xArray = xArrayTemp[~np.isnan(xArrayTemp)]
    if sortOpt == 0:
        xSort = np.sort(xArray)
    elif sortOpt == 1:
        xSort = np.sort(xArray)[::-1]
    elif sortOpt == -1:
        xSort = xArray
    return (xSort)


def plotMap(data, ax=None, lat=None, lon=None, title=None, figsize=(8, 4),
            clbar=True, cmap=plt.cm.jet, bounding=None, prj='cyl'):
    data_sort = flatData(data)
    vmin, vmax = np.percentile(data_sort, 5), np.percentile(data_sort, 95)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
    if bounding is None:
        bounding = [np.min(lat)-0.5, np.max(lat)+0.5, np.min(lon)-0.5, np.max(lon)+0.5]

    m = basemap.Basemap(llcrnrlat=bounding[0], urcrnrlat=bounding[1], llcrnrlon=bounding[2], urcrnrlon=bounding[3],
                         projection=prj, resolution='c', ax=ax)
    # m.drawcoastlines()
    # m.drawstates(linestyle='dashed')
    # m.drawcountries(linewidth=1.0, linestyle='-.')

    m.drawparallels(range(25, 53, 5), labels=[1, 0, 0, 0], fontsize=10)  # 显示经纬度
    m.drawmeridians(range(-125, -66, 10), labels=[0, 0, 0, 1], fontsize=10)

    x, y = m(lon, lat)
    # cs = m.scatter(x, y, c=data, s=30, marker='s', cmap=cmap, vmin=vmin, vmax=vmax)
    cs = m.imshow(data, cmap='RdYlGn', interpolation='bilinear', alpha=0.6, origin='upper')

    if clbar is True:
        # m.colorbar(cs, pad='5%', location='bottom', extend='both')
        m.colorbar(cs, pad='5%', location='right', extend='neither')
    if title is not None:
        ax.set_title(title)

    return fig, ax, m


## exp_design 部分，0.5参数对比，两张图。
def _plot_fig1():
    root = Path(__file__).absolute().parent.parent
    # df_latlon = pd.read_csv(os.path.join(root, "data_dPL/deg0.5/crd.csv"), names=['lat', 'lon'])
    # print(df_latlon)
    # df_params = pd.read_csv(os.path.join(root, "data_dPL/deg0.5/params.csv"))  # 4783 rows x 22 columns
    # print(df_params)
    # var_param = 'ds'
    # title = "ds distribution of dPL-data"
    # save_file = "utils/dPL_ds.pdf"

    df_latlon = pd.read_csv(os.path.join(root, "data_surro/deg0.5/surro_crd_0.5.csv"))
    df_params = pd.read_csv(os.path.join(root, "data_surro/deg0.5/params_sampling1.csv"))
    var_param = 'Ds'
    title = "ds distribution of surro-0.5"
    save_file = "utils/surro-0.5_ds.pdf"

    fig, ax, m = plotMap(data=df_params[var_param].values,
                        lat=df_latlon['lat'].values,
                        lon=df_latlon['lon'].values,
                        title=title)
    fig.savefig(os.path.join(root, save_file), dpi=300)
    plt.show()


## surro-GT某矩形区域结果对比，4宫格
def _plot_fig2():
    root = Path(__file__).absolute().parent.parent
    df_latlon_surro = pd.read_csv(os.path.join(root, "data_surro/deg0.5/surro_crd.csv"))
    df_latlon_rec = pd.read_csv(os.path.join(root, "data_surro/surro_rectangle/surro_crd_rectangle.csv"))

    data_surro = np.load(os.path.join(root, "data_surro/forcing/data_2020.npy"))  # (78496, 366, 8)
    data_rec = np.load(os.path.join(root, "data_surro/surro_rectangle/data_2020.npy"))  # (6400, 366, 8)


    ## 裁剪的更小范围20x20
    data_rec_new = []
    latlon_rec_new = []
    for i in range(20):
        data_rec_new.append(data_rec[i*80:i*80 + 20])
        latlon_rec_new.append(df_latlon_rec.to_numpy()[i*80:i*80 + 20])
    data_rec_new = np.array(data_rec_new)
    latlon_rec_new = np.array(latlon_rec_new)
    print(data_rec_new.shape, latlon_rec_new.shape) #(20, 20, 366, 8) (20, 20, 2)
    fig, ax, m = plotMap(data=data_rec_new[:, :, 0, -1],
                         lat=latlon_rec_new[:, :, 0].flatten(),
                         lon=latlon_rec_new[:, :, 1].flatten())
    plt.show()




if __name__ == '__main__':
    root = Path(__file__).absolute().parent.parent

    # _plot_fig1()
    _plot_fig2()
























