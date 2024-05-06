import numpy as np
import xarray as xr

from windtools.io.binary import BinaryFile


def get_delta(da,assert_equal=True,assert_increasing=True):
    arr = da.values
    diff = np.diff(arr)
    if assert_equal:
        assert np.all(diff[0] == diff)
    if assert_increasing:
        assert np.all(diff > 0)
    return diff[0]


class HAWCInput(object):

    def __init__(self,
        URef=1.0, # dt==dx
        RefHt=150.0, # vertical center of grid [m]
        TipRad=120.97, # distance from rotor apex to blade tip [m]
    ):
        self.URef = URef
        self.RefHt = RefHt
        self.TipRad = TipRad


    def write_binary_from_netcdf(self,ds,
                                 prefix='./',
                                 dy=None,dz=None,
                                 expand_rotor_radius=1.1):
        """Write wind data from netcdf dataset, assuming that the
        WindProfile type is 0 (constant)

        Expected data vars:
        - u
        - v
        - w (optional)
        and coordinates:
        - t
        - z
        - y (optional)
        """
        assert all([varn in ds.coords for varn in ['t','z']])
        assert all([varn in ds.data_vars for varn in ['u','v']])

        #
        # setup rotor grid
        #
        t = ds.coords['t']
        nt = len(t)
        dt = get_delta(t)
        dx = dt * self.URef
        # by default, use same grid spacing as the input coordinates
        if dz is None:
            dz = get_delta(ds.coords['z'])
        if (dy is None) and ('y' in ds.coords):
            dy = get_delta(ds.coords['y'])

        # expand the rotor inflow plane to allow for rotor motion
        nspan = int(np.ceil(expand_rotor_radius * self.TipRad / dz))
        nz = 2*nspan + 1
        z = np.linspace(-nspan*dz, nspan*dz, nz) + self.RefHt
        if dy is None:
            y = np.array([-nspan,nspan])*dz
            ny = 2
        else:
            ny = nz
            y = np.linspace(-nspan*dy, nspan*dy, ny)

        #
        # output params
        #
        self.nx = nt
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy if (dy is not None) else y[1]-y[0]
        self.dz = dz
        self.y = y
        self.z = z
        print('nx,ny,nz =',self.nx,self.ny,self.nz)
        print('dx,dy,dz =',self.dx,self.dy,self.dz)
        print('ygrid =',self.y)
        print('zgrid =',self.z)

        #
        # write inflow data
        #
        ufile = f'{prefix}u.bin'
        vfile = f'{prefix}v.bin'
        wfile = f'{prefix}w.bin'
        interp_coords = {'z': z}

        with BinaryFile(ufile,'w') as f:
            # last plane of turbulence box enters rotor first, and corresponds to
            # the first time snapshot
            for i in range(nt): # indexing goes nx, nx-1, ... 1
                for j in range(ny)[::-1]: # indexing goes ny, ny-1, ... 1
                    if dy is not None:
                        interp_coords['y'] = y[j]
                    # InflowWind will add URef back to the x-component
                    udata = ds['u'].isel(t=i).interp(interp_coords).values - self.URef
                    assert len(udata) == nz, \
                        f'len(interp(u))={len(udata)} (expected nz={nz})'
                    f.write_float(udata) # indexing goes 1, 2, ... nz
            print('Wrote binary',ufile)

        with BinaryFile(vfile,'w') as f:
            # last plane of turbulence box enters rotor first, and corresponds to
            # the first time snapshot
            for i in range(nt): # indexing goes nx, nx-1, ... 1
                for j in range(ny)[::-1]: # indexing goes ny, ny-1, ... 1
                    if dy is not None:
                        interp_coords['y'] = y[j]
                    vdata = ds['v'].isel(t=i).interp(interp_coords).values
                    assert len(vdata) == nz, \
                        f'len(interp(v))={len(vdata)} (expected nz={nz})'
                    f.write_float(vdata) # indexing goes 1, 2, ... nz
            print('Wrote binary',vfile)

        with BinaryFile(wfile,'w') as f:
            if 'w' in ds.data_vars:
                # last plane of turbulence box enters rotor first, and corresponds to
                # the first time snapshot
                for i in range(nt): # indexing goes nx, nx-1, ... 1
                    for j in range(ny)[::-1]: # indexing goes ny, ny-1, ... 1
                        if dy is not None:
                            interp_coords['y'] = y[j]
                        wdata = ds['w'].isel(t=i).interp(interp_coords).values
                        assert len(wdata) == nz, \
                            f'len(interp(w))={len(wdata)} (expected nz={nz})'
                        f.write_float(wdata) # indexing goes 1, 2, ... nz
            else:
                # all 0
                f.write_float(np.zeros((nt,ny,nz)).ravel())
            print('Wrote binary',wfile)


    def write_hourly_binary_from_netcdf(self,ds,prefix='./'):
        """Break up inflow time history (from netcdf file) into steady-
        state runs for each hour. `TMax` is the simulation length for
        each steady-state run.
        """
        t = ds.coords['t'].values
        for ihr,ti in enumerate(np.arange(0,t[-1],3600)):
            # winds are constant for the selected hour
            dshr = ds.interp(t=ti)
            dshr1 = dshr.assign_coords(t=[ti+3600])
            dshr = xr.concat([dshr,dshr1], 't')
            self.write_binary_from_netcdf(dshr, prefix=prefix+f'_{ihr:02d}_')

