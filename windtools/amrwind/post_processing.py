import pandas as pd
import xarray as xr
import numpy as np
import os
import dask.array as da
import h5py
from netCDF4 import Dataset

class ABLStatistics(object):

    def __init__(self,fpath,start_date=None,mean_profiles=False,
                 calc_TI=False,calc_TI_TKE=False):
        """Load planar averaged ABL statistics into an underlying xarray
        dataset for analysis. By default, only surface time histories
        are loaded; setting `mean_profiles=True` will also load
        time--height data from the "mean_profiles" group within the
        dataset.

        Turbulence intensities can be estimated with the `calc_TI` or
        `calc_TI_TKE` parameters. The former is based on horizontal
        variances only (<u'u'>, <u'v'>, and <v'v'>) and accounts for
        flow directionality, whereas the latter is based on an average
        of three velocity variances (<u'u'>, <v'v'>, and <w'w'>).
        """
        self._check_fpaths(fpath)
        if start_date:
            self.datetime0 = pd.to_datetime(start_date)
        else:
            self.datetime0 = None
        self._load_timeseries(mean_profiles)
        self.t = self.ds.coords['time']
        if mean_profiles:
            self.z = self.ds.coords['height']
            self._calc_total_fluxes()
            if calc_TI: self._calc_TI()
            if calc_TI_TKE: self._calc_TI_TKE()

    def _check_fpaths(self,fpath):
        assert isinstance(fpath, (str,list,tuple))
        if isinstance(fpath, str):
            fpath = [fpath]
        self.fpaths = fpath
        for fpath in self.fpaths:
            assert os.path.isfile(fpath), f'{fpath} not found'

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

    def _load_timeseries(self,load_mean_profiles=False):
        dslist = []
        for fpath in self.fpaths:

            ds = xr.load_dataset(fpath)
            ds = self._setup_time_coords(ds)

            if load_mean_profiles:
                pro = xr.load_dataset(fpath, group='mean_profiles')
                pro = pro.rename({'h':'height'})
                pro = pro.assign_coords({
                    self.time_coord: ('num_time_steps',
                                      ds.coords[self.time_coord].values),
                    'height': pro['height'],
                })
                pro = pro.swap_dims({'num_time_steps':self.time_coord,
                                     'nlevels':'height'})
                # make sure underlying array data have the expected shape
                pro = pro.transpose(self.time_coord,'height')
                # merge time-height profiles with timeseries
                ds = xr.combine_by_coords([ds, pro])

            dslist.append(ds)

        if len(dslist) > 1:
            for ds,fpath in zip(dslist,self.fpaths):
                # conflicting attrs causes a merge error
                print('Merging', fpath, ds.attrs.pop('created_on').rstrip())
            # concat and merge
            self.ds = xr.combine_by_coords(dslist)
        else:
            self.ds = dslist[0]

    def _calc_total_fluxes(self):
        for varn in self.ds.data_vars:
            if varn.endswith('_r'):
                varn_sfs = varn[:-2] + '_sfs'
                if varn_sfs in self.ds.data_vars:
                    varn_tot = varn[:-2] + '_tot'
                    self.ds[varn_tot] = self.ds[varn] + self.ds[varn_sfs]

    def _calc_TI(self):
        ang = np.arctan2(self.ds['v'], self.ds['u'])
        rotatedvar = self.ds["u'u'_r"] * np.cos(ang)**2 \
                   + self.ds["u'v'_r"] * 2*np.sin(ang)*np.cos(ang) \
                   + self.ds["v'v'_r"] * np.sin(ang)**2
        self.ds['TI'] = np.sqrt(rotatedvar) / self.ds['hvelmag']

    def _calc_TI_TKE(self):
        meanvar = (self.ds["u'u'_r"] + self.ds["v'v'_r"] + self.ds["w'w'_r"]) / 3
        self.ds['TI_TKE'] = np.sqrt(meanvar) / self.ds['hvelmag']

    def __getitem__(self,key):
        return self.ds[key]

    def rolling_mean(self,Tavg,resample=False,resample_offset='1s'):
        """Calculate a rolling mean assuming a fixed time-step size.
        The rolling window size Tavg is given in seconds.
        """
        dt = float(self.t[1] - self.t[0])
        if not resample:
            assert np.all(np.diff(self.t) == dt), \
                    'Output time interval is variable, set resample=True'
        elif not np.all(np.diff(self.t) == dt):
            if self.datetime0 is None:
                print('Converting to TimedeltaIndex')
                tdelta = pd.to_timedelta(self.ds.coords['time'], unit='s')
                self.ds = self.ds.assign_coords(time=tdelta)
            print('Resampling to',resample_offset,'intervals')
            self.ds = self.ds.resample(time=resample_offset).interpolate()
            self.t = self.ds.coords['time'] / np.timedelta64(1,'s') # convert to seconds
            dt = float(self.t[1] - self.t[0])
        Navg = int(Tavg / dt)
        assert Navg > 0
        return self.ds.rolling(time=Navg).mean()


class Sampling(object):

    def __init__(self,fpath,start_date=None,read_data=False):
        self.fpath = fpath
        self.dt    = None

        self.groups = self._get_groups()
        
        if read_data:
            self.read_data(groups_to_read = self.groups)

    def __repr__(self):
        header=f'Following groups sampled:\n'
        return header+''.join([f'    {g}\n' for g in self.groups])


    def getGroupProperties_xr(self, ds=None, group=None):

        if ds is None and group is None:
            raise ValueError(f'Either `ds` or `group` must be specified')

        if ds is None and group is not None:
            ds = xr.open_dataset(self.fpath, group=group, engine='netcdf4')

        self.sampling_type = ds.sampling_type

        [self.nx, self.ny, self.nz] = ds.ijk_dims
        self.ndt = len(ds.num_time_steps)
        self.tdi = ds.num_time_steps[0]
        self.tdf = ds.num_time_steps[-1]

        # Get axes
        self.x = np.sort(np.unique(ds['coordinates'].isel(ndim=0)))
        self.y = np.sort(np.unique(ds['coordinates'].isel(ndim=1)))
        self.z = np.sort(np.unique(ds['coordinates'].isel(ndim=2)))

        # identify the normal
        #if ds.axis3[0] == 1: self.normal='x'
        #if ds.axis3[1] == 1: self.normal='y'
        #if ds.axis3[2] == 1: self.normal='z'



    def getGroupProperties_h5py(self, ds=None, group=None):
        
        if ds is None and group is None:
            raise ValueError(f'Either `ds` or `group` must be specified')
        if ds is None and group is not None:
            ds = h5py.File(self.fpath)[group]


        # Get axes
        self.x = np.sort(np.unique(ds['coordinates'][()][:,0]))
        self.y = np.sort(np.unique(ds['coordinates'][()][:,1]))
        self.z = np.sort(np.unique(ds['coordinates'][()][:,2]))

        # Get sampling type
        self.sampling_type = ds.attrs['sampling_type'].decode('UTF-8')
        # Get number of points
        [self.nx, self.ny, self.nz] = len(self.x), len(self.y), len(self.z)
        # Get time step information
        firstvar = list(ds.keys())[2]  # likely 'velocityx`, but will be something else if velocity isn't sampled
        self.ndt = ds[firstvar].shape[0]
        self.tdi = 0
        self.tdf = self.ndt-1

        # Identify the normal
        # if self.ny == 1: self.normal='y'
        # if self.nz == 1: self.normal='z'
        # if self.nz == 1: self.normal='z'


    def _get_groups(self):
        with Dataset(self.fpath) as f:
            groups = list(f.groups.keys())
        groups = [groups] if isinstance(groups,str) else groups
        return groups


    def set_dt(self, dt):
        self.dt = dt

    def read_data(self, groups_to_read):

        groups_to_read = groups_to_read if isinstance(groups_to_read,str) else groups_to_read

        ds_all = []
        for g in groups_to_read:
            ds_single = read_single_group(g)
            ds_all.append(ds_single)

        return ds_all

    def read_single_group(self, group, itime=0, ftime=-1, step=1, outputPath=None, var=['velocityx','velocityy','velocityz'], simCompleted=False, verbose=False, package='xr'):

        if package == 'xr':
            print(f'Reading single group using xarray. This will take longer and require more RAM')
            ds = self.read_single_group_xr(group, itime, ftime, step, outputPath, var, simCompleted, verbose)
        elif package == 'h5py':
            print(f'Reading single group using h5py. This is fast, but future computations might be slow')
            ds = self.read_single_group_h5py(group, itime, ftime, step, outputPath, var, simCompleted, verbose)
        else:
            raise ValueError('Package can only be `h5py` or `xr`.')

        return ds




    def read_single_group_h5py(self, group, itime=0, ftime=-1, step=1, outputPath=None, var=['velocityx','velocityy','velocityz'], simCompleted=False, verbose=False):
        '''

        step: int
            Get the output at every step steps. For instance, if a sampling needs to be done
            at 2 s, and other at 0.5 s, then save everything at 0.5 s and when reading the group 
            related to the 2-s sampling, set step to 4
        outputPath: str (default:None)
            If different than None, it is the directory where intermediate and final (concatenated)
            files will be saved.
        var: str, list of str
            variables to be outputted. By defaul, u, v, w. If temperature and tke are available,
            use either var='all' or var=['velocityx','velocityy','velocityz','tke'], for example.
        simCompleted: bool, default False
            If the simulation is still running, the nc file needs to be open using `load_dataset`. 
            This function does _not_ load the data lazily, so it is prohibitively expensive to use
            it in large cases. The function `open_dataset` does load the data lazily, however, it
            breaks any currently running simulation as it leaves the file open for reading, clashing
            with the code that has it open for writing. If the simulation is done, the file is no
            longer open for writing and can be opened for reading using lazy `open_dataset`. This
            bool variable ensures that you _explicitly_ state the simulation is done, so that you
            don't crash something by mistake. simCompleted set to true will likely need to come with
            the specification of nchunks>1, since memory is likely an issue in these cases.

        '''
        
        if simCompleted:
            dsraw = h5py.File(self.fpath)[group]
        else:
            raise ValueError('Unclear if loading the data on a running sim works with h5py. Proceed with caution.')


        if isinstance(var,str): var = [var]
        if var==['all']:
            self.reqvars = ['velocityx','velocityy','velocityz','temperature','tke']
        else:
            self.reqvars = var

        self.getGroupProperties_h5py(ds=dsraw)

        if   self.sampling_type == 'LineSampler':
            ds = self._read_line_sampler(dsraw)
        elif self.sampling_type == 'LidarSampler':
            ds = self._read_lidar_sampler(dsraw)
        elif self.sampling_type == 'PlaneSampler':
            ds = self._read_plane_sampler_h5py(dsraw, group, itime, ftime, step, outputPath, verbose)
        elif self.sampling_type == 'ProbeSampler':
            ds = self._read_probe_sampler(dsraw)
        else:
            raise ValueError(f'Stopping. Sampling type {self.sampling_type} not recognized')

        return ds




    def read_single_group_xr(self, group, itime=0, ftime=-1, step=1, outputPath=None, var=['velocityx','velocityy','velocityz'], simCompleted=False, verbose=False):
        
        if simCompleted:
            dsraw = xr.open_dataset(self.fpath, group=group, engine='netcdf4')
        else:
            dsraw = xr.load_dataset(self.fpath, group=group, engine='netcdf4')


        if isinstance(var,str): var = [var]
        if var==['all']:
            self.reqvars = ['velocityx','velocityy','velocityz','temperature','tke']
        else:
            self.reqvars = var


        self.getGroupProperties_xr(ds = dsraw)

        if   self.sampling_type == 'LineSampler':
            ds = self._read_line_sampler(dsraw)
        elif self.sampling_type == 'LidarSampler':
            ds = self._read_lidar_sampler(dsraw)
        elif self.sampling_type == 'PlaneSampler':
            ds = self._read_plane_sampler_xr(dsraw, group, itime, ftime, step, outputPath, verbose)
        elif self.sampling_type == 'ProbeSampler':
            ds = self._read_probe_sampler(dsraw)
        else:
            raise ValueError(f'Stopping. Sampling type {self.sampling_type} not recognized')

        return ds


    def _read_line_sampler(self,ds):
        raise NotImplementedError(f'Sampling `LineSampler` is not implemented. Consider implementing it..')

    def _read_lidar_sampler(self,ds):
        raise NotImplementedError(f'Sampling `LidarSampler` is not implemented. Consider implementing it.')



    def _read_plane_sampler_h5py(self, ds, group, itime, ftime, step, outputPath, verbose):

        if ftime == -1:
            ftime = self.ndt

        # Unformatted arrays
        #chunksize = (1,-1)  # (time,space); auto means Dask figures out the value, -1 means entire dim

        #chunksize = (-1,'auto', 'auto', 'auto')  # (time,space); auto means Dask figures out the value, -1 means entire dim
        # ValueError: Chunks and shape must be of the same length/dimension. Got chunks=(-1, 'auto', 'auto', 'auto'), shape=(3600, 863744)

        #chunksize = (-1,'auto')  # (time,space); auto means Dask figures out the value, -1 means entire dim
        timeslice = slice(itime, ftime, step)
        velx_old_all = da.from_array(ds['velocityx'] )[timeslice,:]
        vely_old_all = da.from_array(ds['velocityy'] )[timeslice,:]
        velz_old_all = da.from_array(ds['velocityz'] )[timeslice,:]
        #velx_old_all = da.from_array(ds['velocityx'], chunks=chunksize)[timeslice,:]
        #vely_old_all = da.from_array(ds['velocityy'], chunks=chunksize)[timeslice,:]
        #velz_old_all = da.from_array(ds['velocityz'], chunks=chunksize)[timeslice,:]

        # Number of time steps
        ndt = len(velx_old_all)

        # Shaped arrays
        velx_all = np.transpose(velx_old_all.reshape((ndt, self.nz, self.ny, self.nx)), axes=[0,3,2,1])
        vely_all = np.transpose(vely_old_all.reshape((ndt, self.nz, self.ny, self.nx)), axes=[0,3,2,1])
        velz_all = np.transpose(velz_old_all.reshape((ndt, self.nz, self.ny, self.nx)), axes=[0,3,2,1])

        # Rechunk
        #velx_all = velx_all.rechunk((-1,7,'auto','auto'))
        #vely_all = vely_all.rechunk((-1,7,'auto','auto'))
        #velz_all = velz_all.rechunk((-1,7,'auto','auto'))
        velx_all = velx_all.rechunk((-1,7,128,4), balance=True)
        vely_all = vely_all.rechunk((-1,7,128,4), balance=True)
        velz_all = velz_all.rechunk((-1,7,128,4), balance=True)

        # Get the order of the dimensions. Follows the fact that h5py always have [t, z, y, x] and the 
        # transpose call above, setting into the desired order t, x, y, z
        ordereddims = ['samplingtimestep','x','y','z']

        new_all = xr.DataArray(data = velx_all,
                       dims = ordereddims,
                       coords=dict(
                           x=('x',self.x),
                           y=('y',self.y),
                           z=('z',self.z),
                           samplingtimestep=('samplingtimestep',range(itime, ftime, step)),
                       )
                      )
        new_all = new_all.to_dataset(name='u')
        new_all['v'] = (ordereddims, vely_all)
        new_all['w'] = (ordereddims, velz_all)

        if 'temperature' in list(ds.keys()) and 'temperature' in self.reqvars:
            temp_old_all = da.from_array(dsraw['temperature'], chunks=chunksize)[timeslice,:]
            temp_all = temp_old_all.reshape((ndt, nz, ny, nx))
            new_all['temperature'] = (ordereddims, temp_all)

        if 'tke' in list(ds.keys()) and 'tke' in self.reqvars:
            tke_old_all = da.from_array(dsraw['tke'], chunks=chunksize)[timeslice,:]
            tke_all = tke_old_all.reshape((ndt, nz, ny, nx))
            new_all['tke'] = (ordereddims, tke_all)

        if outputPath is not None:
            if outputPath.endswith('.nc'):
                print(f'Saving {outputPath}')
                new_all.to_netcdf(outputPath)
            elif outputPath.endswith('.zarr'):
                print(f'Saving {outputPath}')
                new_all.to_zarr(outputPath)
            else:
                print(f'Saving {group}.nc')
                new_all.to_netcdf(os.path.join(outputPath,f'{group}.nc'))

        return new_all




    def _read_plane_sampler_xr(self, ds, group, itime, ftime, step, outputPath, verbose):

        if ftime == -1:
            ftime = self.ndt

        # Get velocity info (regardless if asked for)

        # Unformatted arrays
        velx_old_all = ds['velocityx'].isel(num_time_steps=slice(itime, ftime, step)).values
        vely_old_all = ds['velocityy'].isel(num_time_steps=slice(itime, ftime, step)).values
        velz_old_all = ds['velocityz'].isel(num_time_steps=slice(itime, ftime, step)).values
 
        # Number of time steps 
        ndt = len(ds['velocityx'].isel(num_time_steps=slice(itime,ftime,step)).num_time_steps)

        velx_all = np.reshape(velx_old_all, (ndt, self.nz, self.ny, self.nx)).T
        vely_all = np.reshape(vely_old_all, (ndt, self.nz, self.ny, self.nx)).T
        velz_all = np.reshape(velz_old_all, (ndt, self.nz, self.ny, self.nx)).T

        # The order of the dimensions varies depending on the `normal`
        if   (ds.axis3 == [1,0,0]).all(): ordereddims = ['y','z','x','samplingtimestep']
        elif (ds.axis3 == [0,1,0]).all(): ordereddims = ['x','z','y','samplingtimestep']
        elif (ds.axis3 == [0,0,1]).all(): ordereddims = ['x','y','z','samplingtimestep']
        else:
            self._read_nonaligned_plane_sampler_xr(ds)
            ordereddims = ['x','y','z','samplingtimestep']

        new_all = xr.DataArray(data = velx_all, 
                       dims = ordereddims,
                       coords=dict(
                           x=('x',self.x),
                           y=('y',self.y),  
                           z=('z',self.z),
                           samplingtimestep=('samplingtimestep',range(itime, ftime, step)),
                       )
                      )
        new_all = new_all.to_dataset(name='u')
        new_all['v'] = (ordereddims, vely_all)
        new_all['w'] = (ordereddims, velz_all)





        print(f'The following variables are available in this dataset: {list(ds.keys())}')

        for curr_var in set(self.reqvars) - {'velocityx', 'velocityy', 'velocityz'}:
            if curr_var not in list(ds.keys()):
                print(f'Variable {curr_var} not available. Skipping it. Available variables: {list(ds.keys())}')
                continue

            temp_old_all = ds[curr_var].isel(num_time_steps=slice(itime, ftime, step)).values
            temp_all = np.reshape(temp_old_all, (ndt, self.nz, self.ny, self.nx)).T
            new_all[curr_var] = (ordereddims, temp_all)


        #if 'actuator_src_termx' in list(ds.keys()) and 'actuator_src_termx' in self.reqvars:
        #    temp_old_all = ds['actuator_src_termx'].isel(num_time_steps=slice(itime, ftime, step)).values
        #    temp_all = np.reshape(temp_old_all, (ndt, self.nz, self.ny, self.nx)).T
        #    new_all['actuator_src_termx'] = (ordereddims, temp_all)

        #if 'temperature' in list(ds.keys()) and 'temperature' in self.reqvars:
        #    temp_old_all = ds['temperature'].isel(num_time_steps=slice(itime, ftime, step)).values
        #    temp_all = np.reshape(temp_old_all, (ndt, self.nz, self.ny, self.nx)).T
        #    new_all['temperature'] = (ordereddims, temp_all)

        #if 'tke' in list(ds.keys()) and 'tke' in self.reqvars:
        #    tke_old_all = ds['tke'].isel(num_time_steps=slice(itime, ftime, step)).values
        #    tke_all = np.reshape(tke_old_all, (ndt, self.nz, self.ny, self.nx)).T
        #    new_all['tke'] = (ordereddims, tke_all)

        if outputPath is not None:
            if outputPath.endswith('.zarr'):
                print(f'Saving {outputPath}')
                new_all.to_zarr(outputPath)
            elif outputPath.endswith('.nc'):
                print(f'Saving {outputPath}')
                new_all.to_netcdf(outputPath)
            else:
                print(f'Saving {group}.nc')
                new_all.to_netcdf(os.path.join(outputPath,f'{group}.nc'))

        return new_all


    def _read_nonaligned_plane_sampler_xr(self, ds):
        print(f"")
        print(f"| ----------------------------- WARNING ----------------------------- |")
        print(f'|  - Unknown normal plane. Guessing the order of the dimensions       |')
        print(f'|  - Assuming no offsets                                              |')
        print(f'|                                                                     |')
        
        # No guarantees that the arrays self.{x,y,z} computed before are correct in this case
        # It is best to reconstruct the arrays based on what we know about the sampling.
        # The sampling points come a bit wonky; e.g. if we requested 258 points in one dir, 
        # and 128 on the other (that is, still a plane), we will end up with the arrays x and y
        # of length 258. Therefore, here we recontruct x and y as being the two principal axis
        # for the plane based on an estimation of the sampling resolution
        
        # Get limits of the slices
        xmin = ds['coordinates'].isel(ndim=0).min().values
        xmax = ds['coordinates'].isel(ndim=0).max().values
        ymin = ds['coordinates'].isel(ndim=1).min().values
        ymax = ds['coordinates'].isel(ndim=1).max().values
        zmin = ds['coordinates'].isel(ndim=2).min().values
        zmax = ds['coordinates'].isel(ndim=2).max().values

        # Let's try to guess which is the second axis. We can identify that by a nice
        # "round" potential resolution. The extent of each direction is in the var
        # self.{x,y,z}. E.g. if it's in the z direction, then (zmax-zmin)/(len(z)-1)
        # would be a nice round number
        potential_res_x = (xmax-xmin)/(len(self.x)-1)
        potential_res_y = (ymax-ymin)/(len(self.y)-1)
        potential_res_z = (zmax-zmin)/(len(self.z)-1)
        if   potential_res_x.is_integer():
            print(f"|  - It seems like the slice's second axis is in the x direction      |")
            potential_res = potential_res_x
            # Estimate the sampling resolution (assuming second axis is in x direction)
            axis1_length = ((ymax-ymin)**2 + (zmax-zmin)**2)**0.5
            axis2_length = ((xmax-xmin)**2)**0.5
            
        elif potential_res_y.is_integer():
            print(f"|  - It seems like the slice's second axis is in the y direction      |")
            potential_res = potential_res_y
            # Estimate the sampling resolution (assuming second axis is in y direction)
            axis1_length = ((xmax-xmin)**2 + (zmax-zmin)**2)**0.5
            axis2_length = ((ymax-ymin)**2)**0.5
            
        elif potential_res_z.is_integer():
            print(f"|  - It seems like the slice's second axis is in the z direction      |")
            potential_res = potential_res_z
            # Estimate the sampling resolution (assuming second axis is in z direction)
            axis1_length = ((xmax-xmin)**2 + (ymax-ymin)**2)**0.5
            axis2_length = ((zmax-zmin)**2)**0.5
        else:
            print(f"|        UNSURE WHAT DIRECTION IS THE SECOND AXIS OF THIS SLICE       |")
            print(f"|                     COMPLICATED SLICE. STOPPING                     |")
            raise ValueError(f'Not sure how to continue with this plane.')
            
        # Guess the estimate. Based on the fact that the first axis is the non-normal-aligned one
        res_guess = axis1_length/self.nx
        print(f"|  It looks like the sampling resolution is approximately {res_guess:.4f} m   |")
        
        # Approximate the guess
        res_guess = np.round(res_guess)
        print(f"|  Aproximating it to {res_guess:.1f} m. Is that the resolution you expected?    |")
        print(f'|                                                                     |')
        
        # Since we now have the potential resolution assuming at least the second
        # axis (axis2) was normal to one plane, let's compare them
        if res_guess == potential_res:
            print(f"| - Seems like the guessed resolution is correct and consistent with  |")
            print(f"|   the resolution from the second axis (normal either x, y, or z)    |")
        else:
            print(f"| - Seems like the guessed resolution above is NOT consistent with    |")
            print(f"|   the resolution from the second axis (normal either x, y, or z),   |")
            print(f"|   which is {potential_res:.2f} m. PROCEED WITH CAUTION. CHECK DOMAIN AND RESOLUT. |")
        
        print(f'|                                                                     |')
        print(f"|   If these guesses are not accurate, STOP NOW and adjust the code.  |")
        print(f'|                                                                     |')
        
        # Creating the axis arrays (assuming x is first axis, y is second axis)
        self.x = np.linspace(0,(self.nx-1)*res_guess,self.nx)
        self.y = np.linspace(0,(self.ny-1)*res_guess,self.ny)
        self.z = [0]
        
        print(f"| ------------------------------------------------------------------- |")





    def _read_probe_sampler(self,ds):
        raise NotImplementedError(f'Sampling `ProbeSampler` is not implemented. Consider implementing it.')



    def to_vtk(self, dsOrGroup, outputPath, verbose=True, offsetz=0, itime_i=0, itime_f=-1, t0=None, dt=None, vtkstartind=0, terrain=False):
        '''
        Writes VTKs for all time stamps present in ds

        dsOrGroup: DataSet or str
            If given a dataset (obtained before with `read_single_group`) then it uses that. If string
            is given, then calls the `read_single_group` before proceeding.
        outputPath: str
            Path where the VTKs should be saved. Should exist. This is useful when specifying 'Low' and
            'HighT*' high-level directories. outputPath = os.path.join(path,'processedData','HighT1')
        offsetz: scalar
            Offset in the z direction, used to make sure both high- and low-res boxes have a point in the
            desired height. It is needed, e.g., when there is a cell edge at 150, and cell center at 145,
            and we are interested in having a cell center at 150. This variable simply shifts everything
            by offsetz, leaving us with points in the height we want. On 10-m grid, offsetz=5 for hh=150;
            and on a 2.5m grid, offsetz=1.25 for hh=150m. It is typically the origin in z for AMR-Wind
            sampling.
        itime_i, itime_f: int
            Initial and final index for time if only a subset of the dataset is desired
        t0: int, float
            The time corresponding to the first saved time step, in s (e.g. t0=20000)
        dt: int, float
            Time step of the underlying data, in seconds. If not none, the files created will have the
            actual time.  If None, the files created follow FAST.Farm convention of time step numbering
        vtkstartind: int
            Index by which the names of the vtk files will be shifted. This is useful for saving files
            starting from a non-zero time-step when AMR-Wind crashes unxpectedly and is restarted using
            a savefile.
        terrain: bool
            Whether or not put NaNs where the terrain is. For this option to be enabled, the dataset
            should also contain a variable called `terrainBlank` taking the value of 1 when it's terrain
            and 0 when it's not. This variable will dictate the velocity values that will become NaNs.

        '''
        if not os.path.exists(outputPath):
            raise ValueError(f'The output path should exist. Stopping.')

        if terrain:
            var = ['velocityx','velocityy','velocityz','terrainBlank']
        else:
            var = ['velocityx','velocityy','velocityz']

        if isinstance(dsOrGroup,xr.Dataset):
            ds = dsOrGroup
            ndt = len(ds.samplingtimestep)
            xarray = ds.x
            yarray = ds.y
            zarray = ds.z
        else:
            if verbose: print(f'    Reading group {dsOrGroup}, from sampling time step {itime_i} to {itime_f}...', flush=True)
            ds = self.read_single_group(dsOrGroup, itime=itime_i, ftime=itime_f, outputPath=None,
                                        simCompleted=True, verbose=verbose, var=var)
            if verbose: print(f'    Done reading group {dsOrGroup}, from sampling time steps above.', flush=True)
            ndt = len(ds.samplingtimestep)  # rt mar13: I had ds.num_time_steps here, but it kept crashing 
            xarray = ds.x
            yarray = ds.y
            zarray = ds.z

        if terrain:
            ds['u'] = ds['u'].where(ds['terrainBlank'] == 0, np.nan)
            ds['v'] = ds['v'].where(ds['terrainBlank'] == 0, np.nan)
            ds['w'] = ds['w'].where(ds['terrainBlank'] == 0, np.nan)

        if t0 is None and dt is None:
            timegiven=False
        else:
            timegiven=True
            if dt%0.1 > 1e10:
                # If dt has two or more significant digits, then a change in the code is required.
                raise ValueError (f'dt with 2 significant digits requires a change in the code')
            if t0 is not None and dt is None or t0 is None and dt is not None:
                raise ValueError (f'If specifying the time, both t0 and dt need to be given')
            if t0<=0 or dt<=0:
                raise ValueError (f'Both the dt and t0 need to be positive')

        if itime_f==-1:
            itime_f = ndt


        for t in np.arange(itime_i, itime_f):

            dstime = ds.sel(samplingtimestep=t)
            if timegiven:
                currentvtk = os.path.join(outputPath,f'Amb.time{t0+t*dt:1f}s.vtk')
            else:
                currentvtk = os.path.join(outputPath,f'Amb.t{vtkstartind+t}.vtk')

            if verbose:
                print(f'Saving {currentvtk}', flush=True)

            with open(currentvtk,'w', encoding='utf-8') as vtk:
                vtk.write(f'# vtk DataFile Version 3.0\n')
                if timegiven:
                    vtk.write(f'{self.sampling_type} corresponding to time {t0+t*dt} s with offset in z of {offsetz}\n')
                else:
                    vtk.write(f'{self.sampling_type} with offset in z of {offsetz}\n')
                vtk.write(f'ASCII\n')
                vtk.write(f'DATASET STRUCTURED_POINTS\n')
                vtk.write(f'DIMENSIONS {self.nx} {self.ny} {self.nz}\n')
                vtk.write(f'ORIGIN {self.x[0]} {self.y[0]} {self.z[0]+offsetz}\n')
                vtk.write(f'SPACING {self.x[1]-self.x[0]} {self.y[1]-self.y[0]} {self.z[1]-self.z[0]}\n')
                vtk.write(f'POINT_DATA {self.nx*self.ny*self.nz}\n')
                vtk.write(f'FIELD attributes 1\n')
                vtk.write(f'U 3 {self.nx*self.ny*self.nz} float\n')
                
                # Read the all u,v,w values in a single function call
                point = dstime.sel(x=xarray,y=yarray,z=zarray)
                
                # Reshape the data to get it in an order required by FAST.FARM
                uval = np.array(point.u.values)
                vval = np.array(point.v.values)
                wval = np.array(point.w.values)
                uval = uval.transpose((2,1,0)).reshape(-1)
                vval = vval.transpose((2,1,0)).reshape(-1)
                wval = wval.transpose((2,1,0)).reshape(-1)
                
                # Write the reshaped numpy array to the file
                np.savetxt(vtk,np.stack((uval,vval,wval)).transpose(),fmt='%.5f',delimiter='\t',newline='\n',encoding='utf-8')

        return ds

def addDatetime(ds,dt,origin=pd.to_datetime('2000-01-01 00:00:00'), computemean=True):
    '''
    Add temporal means to dataset. Means are computed for every spatial location
    separately. That is, no spatial average is used to compute temporal mean. 
    '''
    
    if dt <= 0:
        raise ValueError(f'The dt should be positive. Received {dt}.')

    # Add time array
    ds['time'] = (('samplingtimestep'), ds['samplingtimestep'].values*dt)

    # Save original sampling time step array
    samplingtimestep = ds['samplingtimestep'].values

    # Rename and add datetime information
    ds = ds.rename({'samplingtimestep':'datetime'})
    ds = ds.assign_coords({'datetime':pd.to_datetime(ds['time'], unit='s', origin=origin)})

    # Add back the original sampling time step
    ds['samplingtimestep'] = (('datetime'), samplingtimestep)

    if computemean:
        # Compute or grab means (mean will be available if chunked saving)
        if 'umean' in ds.keys():
            meanu = ds['umean']
            meanv = ds['vmean']
            meanw = ds['wmean']
            ds = ds.drop_vars(['umean','vmean','wmean'])
        else:
            meanu = ds['u'].mean(dim='datetime')
            meanv = ds['v'].mean(dim='datetime')
            meanw = ds['w'].mean(dim='datetime')

        # Add mean computations
        ds['up'] = ds['u'] - meanu
        ds['vp'] = ds['v'] - meanv
        ds['wp'] = ds['w'] - meanw

    return ds
