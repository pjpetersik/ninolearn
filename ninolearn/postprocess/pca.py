import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from sklearn.decomposition.pca import PCA
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
from scipy.signal import detrend

from ninolearn.IO.read_post import data_reader
from ninolearn.pathes import postdir
from ninolearn.utils import generateFileName, scaleMax
from ninolearn.plot.nino_timeseries import nino_background


class pca(PCA):
    """
    This class extends the PCA class of the sklearn.decomposition.pca modlue.
    It facilitates the loading of the data from the postprocessed directory,
    wraps the fit function of the PCA class, has a saving routine for the
    computed pca component and can plot the EOF to get more insight into the
    results.
    """
    def load_data(self, variable, dataset, processed='anom',
                  startyear=1949, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30):
        """
        Load data for PCA analysis from the desired postprocessed data set

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
        grid for which the metrics shell be computed (from 0 to 360 degrees
        east)

        :param lat_min,lat_max:the min and the max values of the latitude grid
        for which the metrics shell be computed (from -180 to 180 degrees east)

        :type plot: bool
        :param: make some plots to view the EOFs
        """

        self.variable = variable
        self.dataset = dataset
        self.processed = processed

        self.startdate = pd.to_datetime(str(startyear))
        self.enddate = (pd.to_datetime(str(endyear)) +
                        pd.tseries.offsets.YearEnd(0))

        self.reader = data_reader(startdate=self.startdate,
                                  enddate=self.enddate,
                                  lon_min=lon_min, lon_max=lon_max,
                                  lat_min=lat_min, lat_max=lat_max)

        data = self.reader.read_netcdf(variable, dataset, processed)

        self.set_eof_array(data)

    def set_eof_array(self, data):
        """
        Genrates the array that will be analyzed with the EOF.
        """
        self.time = data['time']
        self.lon = data['lon']
        self.lat = data['lat']

        EOFarr = np.array(data[:, :, :])

        self.n_time = len(self.time)
        self.n_lat = len(self.lat)
        self.n_lon = len(self.lon)
        self.nan_index = np.isnan(EOFarr[-1,:,:])

        self.EOFarr = EOFarr.reshape((self.n_time,
                                      self.n_lat * self.n_lon))


        self.EOFarr[np.isnan(self.EOFarr)] = 0
        self.EOFarr = detrend(self.EOFarr, axis=0)


    def compute_pca(self):
        """
        Simple wrapper around the PCA.fit() method.
        """
        self.fit(self.EOFarr)

    def save(self, extension='', filename=None):
        """
        Saves the first three pca components to a csv-file.
        """
        # save data to first day of month
        save_index = self.time.to_index()

        pca1 = pd.Series(np.matmul(self.EOFarr, self.components_[0, :]),
                         index=save_index)
        pca2 = pd.Series(np.matmul(self.EOFarr, self.components_[1, :]),
                         index=save_index)
        pca3 = pd.Series(np.matmul(self.EOFarr, self.components_[2, :]),
                         index=save_index)

        self.df = pd.DataFrame({'pca1': pca1, 'pca2': pca2, 'pca3': pca3})

        if filename is None:
            filename = generateFileName(self.variable, self.dataset,
                                    ''.join((self.processed, extension)),
                                    suffix='csv')

        else:
            filename = '.'.join((filename,'csv'))

        filename = '-'.join(['pca', filename])

        self.df.to_csv(join(postdir, filename))

    def plot_eof(self):
        """
        Make a plot for the first leading EOFs.
        """
        lon2, lat2 = np.meshgrid(self.lon, self.lat)

        try:
            nino34 = self.reader.read_csv('nino3.4S')
        except IndexError:
            """
            allow error when  data is out of range for ONI index
            """
            pass
        except AttributeError:
            """
            allow when now reader was initialized (data was not loaded but directly)
            provided with the .set_eof_array() method
            """
            pass

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
            cs = m.pcolormesh(x, y, scaleMax(
                              self.components_[i, :].reshape(self.n_lat,
                                                             self.n_lon)),
                              cmap=cmap, norm=norm)
            m.colorbar(cs)

        for i in range(0, 2):
            fig.add_subplot(223+i)
            projection = np.matmul(self.EOFarr, self.components_[i, :])
            try:
                nino_background(nino34)
            except:
                pass
            plt.plot(self.time, projection)

    def component_map_(self, eof=1):
        """
        Returns the components as a map.

        :param eof: The leading eof (default:1).
        """
        comp_map = self.components_[eof-1, :].copy().reshape(self.n_lat, self.n_lon)
        comp_map[self.nan_index] = np.nan
        return comp_map

    def pc_projection(self, eof=1):
        """
        Returns the amplitude timeseries of the specified eof.

        :param eof: The nth leading eof (default:1).
        """
        projection = np.matmul(self.EOFarr, self.components_[eof-1, :])
        return projection
