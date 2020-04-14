# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import os
import struct

def to_InflowWind(ds, outdir):
    """Write out Binary HAWC-Style Full-Field Files

    From the InflowWind manual:
    ```
    HAWC FF files are 3-dimensional data sets (functions of x, y, and z) of the
    3-component wind inflow velocity, u, v, w. The data are stored in a nx × ny
    × nz evenly-spaced grid, V(x,y,z).
    
    HAWC-style binary files do not contain any header information. All data
    necessary to read and scale it must be entered in the InflowWind input file.
    Each data file contains the wind speed for a specific wind component, stored
    as 4-byte real numbers.
    ```

    These were selected as the preferred input format due to the format
    simplicity. This function generates u.bin, v.bin, and w.bin from an xarray
    Dataset with dimensions 't', 'y', and 'z' in a turbine frame of reference
    (x is streamwise, z is up) and with variables 'u','v', and 'w' corresponding
    to the velocity components.
    """
    Nt = ds.dims['t']
    Ny = ds.dims['y']
    Nz = ds.dims['z']
    fmtstr = '{:d}f'.format(Nz)
    for varname in ['u','v','w']:
        fpath = os.path.join(outdir, varname+'.bin')
        with open(fpath,'wb') as f:
            data = ds[varname].values
            # file format (from InflowWind manual):
            #
            #   ix = nx, nx-1, ... 1
            #       iy = ny, ny-1, ... 1
            #           iz = 1, 2, ... nz
            #               Vgrid(iz,iy,ix,i)
            #           end iz
            #       end iy
            #   end ix
            #
            # where looping over ix can be replaced with
            #     it = 1, 2, ... nt
            # because the last (nx) plane in the inflow box corresponds to the 
            # first inflow timestep.
            for i in range(Nt):
                for j in range(Ny)[::-1]:
                    f.write(struct.pack(fmtstr, *data[i,j,:]))
        print('Wrote',fpath)

