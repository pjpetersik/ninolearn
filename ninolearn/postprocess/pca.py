import pandas as pd
import numpy as np
from os.path import join
from sklearn.decomposition.pca import PCA

from ninolearn.IO.read_post import data_reader
from ninolearn.pathes import postdir
from ninolearn.utils import generateFileName


def pca_component(variable, dataset, processed='anom',
                  startyear=1948, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30):

    """
    Compute the PCA component for a given variable from the desired
    postprocessed data set

    :type variable: str
    :param variable: the variable for which the network time series should
    be computed

    :type dataset: str
    :param dataset: the dataset that should be used to build the network

    :type processed: str
    :param processed: either '','anom' or 'normanom'

    :param startyear: the first year for which the network analysis should
    be done

    :param endyear: the last year for which the network analysis should be
    done

    :param lon_min,lon_max: the min and the max values of the longitude
    grid for which the metrics shell be computed (from 0 to 360 degrees east)

    :param lat_min,lat_max:the min and the max values of the latitude grid
    for which the metrics shell be computed (from -180 to 180 degrees east)
    """

    startdate = pd.to_datetime(str(startyear))
    enddate = pd.to_datetime(str(endyear)) + pd.tseries.offsets.YearEnd(0)

    reader = data_reader(startdate=startdate, enddate=enddate)
    data = reader.read_netcdf(variable, dataset, processed)

    time = data['time']
    lon = data['lon']
    lat = data['lat']

    EOFarr = np.array(data[:, :, :])

    len_time = len(time)
    len_lat = len(lat)
    len_lon = len(lon)

    EOFarr = EOFarr.reshape((len_time, len_lat * len_lon))

    pca = PCA(n_components=6)
    pca.fit(EOFarr)

    # TODO: save data to first day of month ahead
    save_index = time.to_index()+pd.tseries.offsets.MonthBegin(1)

    pca1 = pd.Series(np.matmul(EOFarr, pca.components_[0, :]),
                     index=save_index)
    pca2 = pd.Series(np.matmul(EOFarr, pca.components_[1, :]),
                     index=save_index)
    pca3 = pd.Series(np.matmul(EOFarr, pca.components_[2, :]),
                     index=save_index)

    df = pd.DataFrame({'pca1': pca1, 'pca2': pca2, 'pca3': pca3})

    filename = generateFileName(variable, dataset, processed, suffix='csv')
    filename = '-'.join(['pca', filename])

    df.to_csv(join(postdir, filename))


pca_component('air', 'NCEP', 'anom', startyear=1948, endyear=2018)


"""
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import matplotlib.pyplot as plt
from ninolearn.plot.nino_timeseries import nino_background
def scaleMax(x):
    return x/np.max(np.abs(x))

if __name__=="__mainn__":
    lon2, lat2 = np.meshgrid(lon, lat)


    nino34 = reader.read_csv('nino34')
    plt.close("all")
    fig = plt.figure(figsize=(15, 7))


    for i in range(0, 2):
        fig.add_subplot(221+i)
        plt.title("EOF"+str(i+1))
        m = Basemap(projection='robin', lon_0=180, resolution='c')
        x, y = m(lon2, lat2)

        m.drawparallels(np.arange(-90., 120., 30.))
        m.drawmeridians(np.arange(0., 360., 60.))
        m.drawmapboundary(fill_color='white')
        m.drawcoastlines()

        norm = cm.colors.Normalize(vmax=-1, vmin=1.)
        cmap = cm.bwr
        cs = m.pcolormesh(x, y,
                          scaleMax(pca.components_[i, :].reshape(len_lat,
                                                                 len_lon)),
                          cmap=cmap, norm=norm)

        cb = m.colorbar(cs)

    for i in range(0, 2):
        fig.add_subplot(223+i)
        projection = np.matmul(EOFarr, pca.components_[i, :])
        nino_background(nino34)
        plt.plot(time, projection)

"""
