from ninolearn.utils import print_header
from ninolearn.preprocess.network import networkMetricsSeries

print_header("Network Metrics")
#
#nms_ssh_godas = networkMetricsSeries('sshg', 'GODAS', processed="anom",
#                           threshold=0.9, startyear=1980, endyear=2018,
#                           window_size=12, lon_min=120, lon_max=280,
#                           lat_min=-30, lat_max=30, verbose=1)
#nms_ssh_godas.computeTimeSeries()

nms_ssh_oras4 = networkMetricsSeries('zos', 'ORAS4', processed="anom",
                           threshold=0.9, startyear=1959, endyear=2017,
                           window_size=12, lon_min=120, lon_max=280,
                           lat_min=-30, lat_max=30, verbose=1)
nms_ssh_oras4.computeTimeSeries()
