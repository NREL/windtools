import pandas as pd
import xarray as xr

class ABLStatistics(object):

    def __init__(self,fpath,start_date=None,mean_profiles=False):
        self.fpath = fpath
        if start_date:
            self.datetime0 = pd.to_datetime(start_date)
        else:
            self.datetime0 = None
        self._load_timeseries()
        if mean_profiles:
            self._load_timeheight_profiles()

    def _setup_time_coords(self,ds):
        if self.datetime0:
            dt = self.datetime0 + pd.to_timedelta(ds['time'], unit='s')
            ds = ds.assign_coords({'datetime':('num_time_steps',dt)})
            ds = ds.swap_dims({'num_time_steps':'datetime'})
            self.time_coord = 'datetime'
        else:
            ds = ds.swap_dims({'num_time_steps':'time'})
            self.time_coord = 'time'
        return ds

    def _load_timeseries(self):
        ds = xr.load_dataset(self.fpath)
        ds = self._setup_time_coords(ds)
        self.ds = ds

    def _load_timeheight_profiles(self):
        ds = xr.load_dataset(self.fpath, group='mean_profiles')
        ds = ds.rename({'h':'height'})
        times = self.ds.coords[self.time_coord].values
        ds = ds.assign_coords({
            self.time_coord: ('num_time_steps',times),
            'height': ds['height'],
        })
        ds = ds.swap_dims({'num_time_steps':self.time_coord, 'nlevels':'height'})
        ds = ds.transpose(self.time_coord,'height')
        self.ds = xr.combine_by_coords([self.ds, ds])


