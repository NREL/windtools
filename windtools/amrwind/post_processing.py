import pandas as pd
import xarray as xr

class ABLStatistics(object):

    def __init__(self,fpath,startdatetime=None):
        self.fpath = fpath
        if startdatetime:
            self.datetime0 = pd.to_datetime(startdatetime)
        else:
            self.datetime0 = None
        self._load_timeseries()
        #self._load_timeheightseries()

    def _load_timeseries(self):
        ds = xr.open_dataset(self.fpath)
        if self.datetime0:
            dt = self.datetime0 + pd.to_timedelta(ds['time'], unit='s')
            ds = ds.assign_coords({'num_time_steps':dt})
            print(ds)
            ds = ds.swap_dims({'num_time_steps':'datetime'})
        else:
            ds = ds.swap_dims({'num_time_steps':'time'})
        self.ds = ds
