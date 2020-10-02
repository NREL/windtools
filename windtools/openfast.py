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

channelnames = {
    'GenPwr_[kW]': 'Generator Power [kW]',
    'GenTq_[kN-m]': 'Generator Torque [kN-m]',
    'RotSpeed_[rpm]': 'Rotor Speed [RPM]',
    'BldPitch1_[deg]': 'Blade 1 Pitch [deg]',
    'BldPitch2_[deg]': 'Blade 2 Pitch [deg]',
    'BldPitch3_[deg]': 'Blade 3 Pitch [deg]',
    'YawPos_[deg]': 'Nacelle Yaw Position [deg]',
}

InflowWind_template = """------- InflowWind v3.01.* INPUT FILE -------------------------------------------------------------------------
Inflow description here.
---------------------------------------------------------------------------------------------------------------
False         Echo           - Echo input data to <RootName>.ech (flag)
{WindType:>11d}   WindType       - switch for wind file type (1=steady; 2=uniform; 3=binary TurbSim FF; 4=binary Bladed-style FF; 5=HAWC format; 6=User defined)
          0   PropagationDir - Direction of wind propagation (meteoroligical rotation from aligned with X (positive rotates towards -Y) -- degrees)
          1   NWindVel       - Number of points to output the wind velocity    (0 to 9)
          0   WindVxiList    - List of coordinates in the inertial X direction (m)
          0   WindVyiList    - List of coordinates in the inertial Y direction (m)
{RefHt:>11f}   WindVziList    - List of coordinates in the inertial Z direction (m)
================== Parameters for Steady Wind Conditions [used only for WindType = 1] =========================
{URef:>11f}   HWindSpeed     - Horizontal windspeed                            (m/s)
{RefHt:>11f}   RefHt          - Reference height for horizontal wind speed      (m)
          0   PLexp          - Power law exponent                              (-)
================== Parameters for Uniform wind file   [used only for WindType = 2] ============================
"unused"      Filename       - Filename of time series data for uniform wind field.      (-)
{RefHt:>11f}   RefHt          - Reference height for horizontal wind speed                (m)
     125.88   RefLength      - Reference length for linear horizontal and vertical sheer (-)
================== Parameters for Binary TurbSim Full-Field files   [used only for WindType = 3] ==============
"unused"      Filename       - Name of the Full field wind file to use (.bts)
================== Parameters for Binary Bladed-style Full-Field files   [used only for WindType = 4] =========
"unused"      FilenameRoot   - Rootname of the full-field wind file to use (.wnd, .sum)
False         TowerFile      - Have tower file (.twr) (flag)
================== Parameters for HAWC-format binary files  [Only used with WindType = 5] =====================
"{hawc_ufile:s}"    FileName_u     - name of the file containing the u-component fluctuating wind (.bin)
"{hawc_vfile:s}"    FileName_v     - name of the file containing the v-component fluctuating wind (.bin)
"{hawc_wfile:s}"    FileName_w     - name of the file containing the w-component fluctuating wind (.bin)
{nx:>11d}   nx             - number of grids in the x direction (in the 3 files above) (-)
{ny:>11d}   ny             - number of grids in the y direction (in the 3 files above) (-)
{nz:>11d}   nz             - number of grids in the z direction (in the 3 files above) (-)
{dx:>11f}   dx             - distance (in meters) between points in the x direction    (m)
{dy:>11f}   dy             - distance (in meters) between points in the y direction    (m)
{dz:>11f}   dz             - distance (in meters) between points in the z direction    (m)
{RefHt:>11f}   RefHt          - reference height; the height (in meters) of the vertical center of the grid (m)
  -------------   Scaling parameters for turbulence   ---------------------------------------------------------
          0   ScaleMethod    - Turbulence scaling method   [0 = none, 1 = direct scaling, 2 = calculate scaling factor based on a desired standard deviation]
          0   SFx            - Turbulence scaling factor for the x direction (-)   [ScaleMethod=1]
          0   SFy            - Turbulence scaling factor for the y direction (-)   [ScaleMethod=1]
          0   SFz            - Turbulence scaling factor for the z direction (-)   [ScaleMethod=1]
          0   SigmaFx        - Turbulence standard deviation to calculate scaling from in x direction (m/s)    [ScaleMethod=2]
          0   SigmaFy        - Turbulence standard deviation to calculate scaling from in y direction (m/s)    [ScaleMethod=2]
          0   SigmaFz        - Turbulence standard deviation to calculate scaling from in z direction (m/s)    [ScaleMethod=2]
  -------------   Mean wind profile parameters (added to HAWC-format files)   ---------------------------------
{URef:>11f}   URef           - Mean u-component wind speed at the reference height (m/s)
         -1   WindProfile    - Wind profile type (0=constant;1=logarithmic,2=power law)
          0   PLExp          - Power law exponent (-) (used for PL wind profile type only)
        0.0   Z0             - Surface roughness length (m) (used for LG wind profile type only)
====================== OUTPUT ==================================================
True          SumPrint     - Print summary data to <RootName>.IfW.sum (flag)
              OutList      - The next line(s) contains a list of output parameters.  See OutListParameters.xlsx for a listing of available output channels, (-)
"Wind1VelX"               X-direction wind velocity at point WindList(1)
"Wind1VelY"               Y-direction wind velocity at point WindList(1)
"Wind1VelZ"               Z-direction wind velocity at point WindList(1)
END of input file (the word "END" must appear in the first 3 columns of this last OutList line)
---------------------------------------------------------------------------------------"""

def to_InflowWind(ds, outdir='.', prefix=''):
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
    dims = ['t','y','z']
    ds = ds.sortby(dims).transpose(*dims)
    Nt = ds.dims['t']
    Ny = ds.dims['y']
    Nz = ds.dims['z']
    fmtstr = '{:d}f'.format(Nz)
    for varname in ['u','v','w']:
        fpath = os.path.join(outdir, prefix+varname+'.bin')
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

