import pandas as pd
import xarray as xr
import numpy as np
import os
from netCDF4 import Dataset

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
        self.ds[self.time_coord] = np.round(self.ds[self.time_coord],5)





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


    def _get_properties(self, ds):
        self.sampling_type = ds.sampling_type

        [self.nx, self.ny, self.nz] = ds.ijk_dims
        self.ndt = len(ds.num_time_steps)
        self.tdi = ds.num_time_steps[0]
        self.tdf = ds.num_time_steps[-1]

        # identify the normal
        if ds.axis3[0] == 1: self.normal='x'
        if ds.axis3[1] == 1: self.normal='y'
        if ds.axis3[2] == 1: self.normal='z'

        # Get axes
        self.x = np.sort(np.unique(ds['coordinates'].isel(ndim=0)))
        self.y = np.sort(np.unique(ds['coordinates'].isel(ndim=1)))
        self.z = np.sort(np.unique(ds['coordinates'].isel(ndim=2)))


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


    def read_single_group(self, group, verbose=False, nchunks=1, chunk_step=1, outputPath=None):
        '''

        nchunks: int
            Number of chunks that the whole sampling netCDF will be split for reading and processing
        chunk_step: int
            Get the output at every chunk_step steps. For instance, if a sampling needs to be done
            at 2 s, and other at 0.5 s, then save everything at 0.5 s and when reading the group 
            related to the 2-s sampling, set chunk_step to 4
        outputPath: str (default:None)
            If different than None, it is the directory where intermediate and final (concatenated)
            files will be saved.

        '''
        
        dsraw = xr.load_dataset(self.fpath, group=group, engine='netcdf4')

        self._get_properties(dsraw)

        if   self.sampling_type == 'LineSampler':
            ds = self._read_line_sampler(dsraw)
        elif self.sampling_type == 'LidarSampler':
            ds = self._read_lidar_sampler(dsraw)
        elif self.sampling_type == 'PlaneSampler':
            ds = self._read_plane_sampler(dsraw, verbose, nchunks, chunk_step, outputPath)
        elif self.sampling_type == 'ProbeSampler':
            ds = self._read_probe_sampler(dsraw)
        else:
            raise ValueError(f'Stopping. Sampling type {self.sampling_type} not recognized')

        return ds


    def _read_line_sampler(self,ds):
        raise NotImplementedError(f'Sampling `LineSampler` is not implemented. Consider implementing it..')

    def _read_lidar_sampler(self,ds):
        raise NotImplementedError(f'Sampling `LidarSampler` is not implemented. Consider implementing it.')

    def _read_plane_sampler(self, ds, verbose, nchunks, chunk_step, outputPath):

        nchunks = 1
        chunk_ndt = int(self.ndt/nchunks)
        chunk_dti_list = [int(i) for i in np.floor(np.linspace(0,self.ndt,nchunks,endpoint=False))]
        chunk_dtf_list = [i+chunk_ndt for i in chunk_dti_list]

        # Accumulate list of intermediate filenames to open all of them with open_mfdataset
        filenamelist = []

        if verbose:
            print(f'Using {nchunks} chunk(s), with {chunk_ndt} time steps each.')
            print(f'The start/end timestep of each chunk is as follows:')
            print(*list(zip(chunk_dti_list,chunk_dtf_list)), sep='\n')


        for chunk_dti in chunk_dti_list: #[:2]:
            chunk_dtf = chunk_dti+chunk_ndt
            
            if verbose:
                print(f'Processing chunk with start/end time step: {chunk_dti}, {chunk_dtf}')

            try:
                # Unformatted arrays
                velx_old_all = ds['velocityx'].isel(num_time_steps=slice(chunk_dti, chunk_dtf, chunk_step)).values
                vely_old_all = ds['velocityy'].isel(num_time_steps=slice(chunk_dti, chunk_dtf, chunk_step)).values
                velz_old_all = ds['velocityz'].isel(num_time_steps=slice(chunk_dti, chunk_dtf, chunk_step)).values

                velx_all = np.reshape(velx_old_all, (chunk_ndt, self.nz, self.ny, self.nx)).T
                vely_all = np.reshape(vely_old_all, (chunk_ndt, self.nz, self.ny, self.nx)).T
                velz_all = np.reshape(velz_old_all, (chunk_ndt, self.nz, self.ny, self.nx)).T

                # The order of the dimensions varies depending on the `normal`
                if   (ds.axis3 == [1,0,0]).all(): ordereddims = ['y','z','x','samplingtimestep']
                elif (ds.axis3 == [0,1,0]).all(): ordereddims = ['x','z','y','samplingtimestep']
                elif (ds.axis3 == [0,0,1]).all(): ordereddims = ['x','y','z','samplingtimestep']
                else: raise ValueError('Unknown normal plane')

                new_all = xr.DataArray(data = velx_all, 
                               dims = ordereddims,
                               coords=dict(
                                   x=('x',self.x),
                                   y=('y',self.y),  
                                   z=('z',self.z),
                                   samplingtimestep=('samplingtimestep',range(chunk_dti, chunk_dtf)),
                               )
                              )
                new_all = new_all.to_dataset(name='u')
                new_all['v'] = (ordereddims, vely_all)
                new_all['w'] = (ordereddims, velz_all)

                if outputPath is not None:
                    # save file
                    filename = os.path.join(outputPath,f'temp_{group}_dt{chunk_dti}_{chunk_dtf-1}.nc')
                    new_all.to_netcdf(filename)
                    filenamelist.append(filename)
            
            except MemoryError as e:
                print(f'    MemoryError: {e}.\n    Try increasing the number of chunks')
                raise

    
        if outputPath is not None:
            # Concat all files and save a single one
            dsall = xr.open_mfdataset(filenamelist)
            dsall.to_netcdf(outputPath,f'{group}_dt{chunk_dti_list[0]}_{chunk_dtf_list[-1]}.nc')
        else:
            dsall = new_all

        # ds['time'] = (('samplingtimestep'),ds.samplingtimestep.values*150)

        return dsall


    def _read_probe_sampler(self,ds):
        raise NotImplementedError(f'Sampling `ProbeSampler` is not implemented. Consider implementing it.')

    
    def to_vtk(self, dsOrGroup, outputPath, verbose=True, offsetz=0, itime_i=0, itime_f=-1):
        '''
        Writes VTKs for all time stamps present in ds

        dsOrGroup: DataSet or str
            If given a dataset (obtained before with `read_single_group`) then it uses that. If string
            is given, then calls the `read_single_group` before proceeding.
        outputPath: str
            Path where the VTKs should be saved. Should exist. This is useful when specifying 'Low' and
            'HighT*' high-level directories. outputPath = os.path.join(path,'processedData','HighT1')
        itime_i, itime_f: int
            Initial and final index for time if only a subset of the dataset is desired

        '''
        if not os.path.exists(outputPath):
            raise ValueError(f'The output path should exist. Stopping.')

        if isinstance(dsOrGroup,xr.Dataset):
            ds = dsOrGroup
            ndt = len(ds.samplingtimestep)
            xarray = ds.x
            yarray = ds.y
            zarray = ds.z
        else:
            ds = self.read_single_group(dsOrGroup)
            ndt = len(ds.num_time_steps)
            xarray = self.x
            yarray = self.y
            zarray = self.z

        if itime_f==-1:
            itime_f = ndt


        for t in np.arange(itime_i, itime_f):

            dstime = ds.isel(samplingtimestep=t)
            currentvtk = os.path.join(outputPath,f'Amb.t{t}.vtk')

            if verbose:
                print(f'Saving {currentvtk}')

            with open(currentvtk,'w', encoding='utf-8') as vtk:
                vtk.write(f'# vtk DataFile Version 3.0\n')
                vtk.write(f'{self.sampling_type} with offset in z of {offsetz}\n')
                vtk.write(f'ASCII\n')
                vtk.write(f'DATASET STRUCTURED_POINTS\n')
                vtk.write(f'DIMENSIONS {self.nx} {self.ny} {self.nz}\n')
                vtk.write(f'ORIGIN {self.x[0]} {self.y[0]} {self.z[0]+offsetz}\n')
                vtk.write(f'SPACING {self.x[1]-self.x[0]} {self.y[1]-self.y[0]} {self.z[1]-self.z[0]}\n')
                vtk.write(f'POINT_DATA {self.nx*self.ny*self.nz}\n')
                vtk.write(f'FIELD attributes 1\n')
                vtk.write(f'U 3 {self.nx*self.ny*self.nz} float\n')
                for z in zarray:
                    for y in yarray:
                        for x in xarray:
                            point = dstime.sel(x=x,y=y,z=z)
                            vtk.write(f'{point.u.values:.5f}\t{point.v.values:.5f}\t{point.w.values:.5f}\n')
            


