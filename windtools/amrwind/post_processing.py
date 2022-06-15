import pandas as pd
import xarray as xr

class ABLStatistics(object):

    def __init__(self,fpath,startdate=None):
        self.fpath = fpath
        if startdate:
            self.datetime0 = pd.to_datetime(startdate)
        else:
            self.datetime0 = None
        self._load_timeseries()
        #self._load_timeheightseries()

    def _setup_time_coords(self,ds):
        if self.datetime0:
            dt = self.datetime0 + pd.to_timedelta(ds['time'], unit='s')
            ds = ds.assign_coords({'datetime':('num_time_steps',dt)})
            ds = ds.swap_dims({'num_time_steps':'datetime'})
        else:
            ds = ds.swap_dims({'num_time_steps':'time'})
        return ds

    def _load_timeseries(self):
        ds = xr.open_dataset(self.fpath)
        ds = self._setup_time_coords(ds)
        self.ds = ds
