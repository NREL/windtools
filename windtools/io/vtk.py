# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""
Functions for handling VTKs
"""

import struct
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from scipy.interpolate import griddata
import xarray as xr
import os
import datetime
import pandas as pd
import numpy as np

def readVTK(vtkpath, sliceType=None, dateref=None, ti=None, tf=None, t=None, res=None, squash=None):
    """
    Read VTK file(s) and interpolate into a uniform grid based on the limits read from the VTK
    and resolution given.

    Function tailored to read multiple files with the following file structure:
        full/path/to/<vtkpath>
        ├── <time1>
        ├── <time2>
        │   ├── <slicetype1>
        │   ├── <slicetype2>
        │   ├── ...
        │   └── <slicetypen>
        ├── ...
        └── <timen>

    To read multiple VTKs, specify sliceType, ti, and tf;
    To read a single VTK, there are two options: (i) specify full vtk path vtkpath, or
        (ii) specity vtkpath, sliceType, and t.

    If reading more than a single VTK, it is assumed that they share the same coordinates.

    The function provides output for full 3-D VTK. If reading planar data, use the squash
    option. 
    
    Example calls:
        # Read multiple
        ds = readVTK('full/path/to/vtks', sliceType='U_zNormal.80.vtk', ti= 133200, tf=133205,
                     dateref= pd.to_datetime('2010-05-14 12:00:00', res=10)
        # Read single
        ds = readVTK('full/path/to/vtks/133200/U_zNormal.80.vtk',
                     dateref= pd.to_datetime('2010-05-14 12:00:00', res=10)
        # Read single
        ds = readVTK('full/path/to/vtks', sliceType='U_zNormal.80.vtk', t= 133200,
                     dateref= pd.to_datetime('2010-05-14 12:00:00', res=10)

    Parameters:
    -----------
    vtkpath: str
        Full path to the directory of time steps containing the VTK files (if reading
        multiple); or full path to the vtk itself, with extension (if reading single)
    sliceType: str
        Common name of the collection of slices to be read. E.g. 'U_zNormal.80.vtk'
    dateref: datetime
        Reference time to be specified if datetime output is desired. Default is
        float/integer output, representing seconds
    ti, tf, t: int, float, str
        Times directories that should be read. If a single time is specified, t, then
        the nearest matching time-directory is read
    res: int, float, or three-component tuple
        resolution of the meshgrid in which the data will be interpolated onto.
        If tuple, provide resolution in x, y, and z, in order
    squash: str; 'x', 'y', or 'z' only
        Squash the VTK into a plane (useful for terrain slices)

    Returns:
    --------
    ds: xr.DataSet
        Dataset containg the data with x, y [, z, time] as coordinates


    written by Regis Thedin (regis.thedin@nrel.gov)
    """


    # Single slice was requested, using `vtlpath='some/path/slice.vtk`
    if os.path.isfile(vtkpath):
        if isinstance(vtkpath, str):
            if vtkpath.endswith('.vtk'):
                print(f'Reading a single VTK. For output with `datetime` as coordinate, specify the path, sliceType, and t (see docstrings for example)')
                x, y, z, out = readSingleVTK(vtkpath, res=res, squash=squash)
                ds = VTK2xarray(x, y, z, out)
                return ds
            else:
                raise SyntaxError("Single vtk specification using vtkpath='/path/to/file.vtk' should include the extension")
        else:
            raise SyntaxError("The vtkpath='path/to/vtks' should be a string.")


    # Some checks
    assert os.path.isdir(vtkpath), f'Directory of VTKs given {vtkpath} is not a directory. For single VTK, see docstrings for example.'
    if not sliceType.endswith('.vtk'):
        raise SyntaxError('sliceType should be given with .vtk extension')
    if ti!=None and tf!=None:
        if t!=None: raise ValueError('You can specify either t only, or ti and tf, but not all three')
        if ti>tf:   raise ValueError('tf should be larger than ti')
        if ti==tf:
            t = ti
    elif tf!= None:
        raise ValueError('If you specify tf, then ti should be specified too')
    elif ti!= None:
        raise ValueError('If you specify ti, then tf should be specified too')
    else:
        if t==None:
            raise ValueError("You have to specify at least one time. If a single VTK is needed, use " \
                             "vtkpath='path/to/file.vtk' ")

    # Get the time directories
    times_str = sorted(os.listdir(vtkpath))
    times_float = [float(i) for i in times_str]

    if t is not None:
        # Single time was requested
        pos_t = min(range(len(times_float)), key=lambda i: abs(times_float[i]-t))
        if pos_t == 0 and t < 0.999*times_float[0]:
            raise ValueError(f'No VTK found for time {t}. The first available VTK is at t={times_float[0]}')
        if pos_t == len(times_str)-1 and t > 1.001*times_float[-1]:
            raise ValueError(f'No VTK found for time {t}. The last available VTK is at t={times_float[-1]}')
        t = times_str[pos_t]
        print(f'Reading a single VTK for time {float(t)}')
        x, y, z, out = readSingleVTK(os.path.join(vtkpath,t,sliceType), res=res, squash=squash)
        ds = VTK2xarray(x, y, z, out, t, dateref)
        return ds

    # Find limits of the requested subset
    pos_ti = min(range(len(times_float)), key=lambda i: abs(times_float[i]-ti))
    pos_tf = min(range(len(times_float)), key=lambda i: abs(times_float[i]-tf))
    nvtk = pos_tf - pos_ti

    if nvtk == 0:
        raise ValueError('No VTKs found for the time range specified. VTKs are available ' \
                        f'between {times_float[0]} and {times_float[-1]}.')

    print(f'Number of VTKs to be read: {nvtk}')

    dslist = []
    for i, t in enumerate(times_str[pos_ti:pos_tf]):
        print(f'Iteration {i}: processing time {t}...  {100*i/nvtk:.2f}%', end='\r', flush=True)
        x, y, z, out = readSingleVTK(os.path.join(vtkpath,t,sliceType), res=res, squash=squash)
        current_ds = VTK2xarray(x, y, z, out, t, dateref)
        dslist.append(current_ds)

    print('\nDone.')

    # Concatenate on the appropriate dimension
    if 'time' in dslist[0].dims:
        ds = xr.concat(dslist, dim='time')
    elif 'datetime' in dslist[0].dims:
        ds = xr.concat(dslist, dim='datetime')

    return ds



def VTK2xarray(x, y, z, out, t=None, dateref=None):
    """
    Convert x, y, z, and out from `readSingleVTK into a dataset.
    If dateref is provided, then datetime is given in the output
    
    If plane of data, then the third direction is ommited and results
    are given in terms of x and y (even if the data in on the yz plane).
    If full 3-D data (no squash), then x, y, and z are given

    Parameters:
    -----------
    x, y, z: array
        Coordinate of the grid
    out: nD-array
        Values at the grid points. Size depends on the number of components
    t: int, float, str (optional)
        time to add to the DataSet as a coordinate
    dateref: datetime (optional)
        Reference time to be specified if datetime output is desired. Requires t
        to be specified as well

    Returns:
    --------
    ds: xr.DataSet
        Dataset containg the data with x, y [, z, time] as coordinates

    written by Regis Thedin (regis.thedin@nrel.gov)
    """

    if dateref is not None and t is None:
        raise SyntaxError('If dateref is specified, t should be specified too')

    if len(out) == 3:
        # vector
        ds = xr.Dataset({'u':(('x','y','z'), out[0,:]), 'v':(('x','y','z'), out[1,:]), 'w':(('x','y','z'), out[2,:])})
    elif len(out) == 1:
        # scalar
        ds = xr.Dataset({'var':(('x','y','z'), out[0,:])})
    elif len(out) == 6:
        # tensor
        ds = xr.Dataset({'var_xx':(('x','y','z'), out[0,:]), 'var_xy':(('x','y','z'), out[1,:]), 'var_xz':(('x','y','z'), out[2,:]), \
                         'var_yy':(('x','y','z'), out[3,:]), 'var_yz':(('x','y','z'), out[4,:]), 'var_zz':(('x','y','z'), out[5,:])})
    else:
        raise ValueError(f'Read variable with {len(out)} components. Not sure how to follow. Stopping.')


    ds = ds.assign_coords({'x':x, 'y':y, 'z':z})

    # If data is on a plane (squash != None), get rid of the third dimesion
    if len(ds.z) == 1:
        ds = ds.squeeze(dim='z').drop_vars('z')
        
    if isinstance(t, (str, int, float)):
        if type(dateref) in (datetime, datetime.datetime, pd.Timestamp):
            t =  pd.to_datetime(float(t), unit='s', origin=dateref).round('0.1S')
            timename = 'datetime'
        else:
            t = float(t)
            timename = 'time'
        ds[timename] = t
        ds = ds.expand_dims(timename).assign_coords({timename:[t]})

    return ds



def readSingleVTK(vtkfullpath, res=None, squash=None):
    """
    Read a single VTK file and interpolates into a uniform grid based on the limits read from the VTK

    If requesting full 3-D field, it may be appropriate to give a tuple (resx, resy, resy) as input in `res`

    For full 3-D field, the function assumes the points are given within a structured grid of
    either uniform resolution `res` or resolution (res[0],res[1],res[2]) in all directions. It uses
    `griddata` for interpolation and thus a very large field will take significant time.

    Parameters:
    -----------
    vtkfullpath: str
        full path and filename of the desired VTK file
    res: int, float, or three-component tuple
        resolution of interpolated grid. If tuple, resolution in x, y, and z, in order
    squash: str, 'x', 'y', or 'z' only
        Squash the VTK into a plane (useful for terrain slices)

    Returns:
    --------
    x, y, z:
        meshgrid of x, y and z positions.
    out:
        array of shape (<numComponents>, <nx>, <ny>, <nz>), where x and y are the long and short 
        direction, respectively. E.g. for Uz, out[2]

    written by Regis Thedin (regis.thedin@nrel.gov)
    """


    # Check if the slice actually exists
    if not os.path.isfile(vtkfullpath):
        raise FileNotFoundError(f'Slice {os.path.basename(os.path.dirname(vtkfullpath))}/' \
                                f'{os.path.basename(vtkfullpath)} does not exist.')

    # Open VTK
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtkfullpath)
    reader.ReadAllVectorsOn()
    reader.Update()

    # Load data
    polydata = dsa.WrapDataObject(reader.GetOutput())
    ptdata   = polydata.GetPointData()
    coords   = np.round(polydata.Points, 4)
    data     = ptdata.GetArray(0)

    # Determine what plane the VTK is in
    if squash == None:
        # Full 3-D field
        dir1=0; dir2=1;
        dirconst=2
        print('Reading full 3D VTK. It might take a while. If you are reading planes of data, use `squash`.')
    elif squash=='x':
        dirconst = 0
        dir1=1;  dir2=2
    elif squash=='y':
        dirconst = 1
        dir1=0;  dir2=2
    elif squash=='z':
        dirconst = 2
        dir1=0; dir2=1;
    else:
        raise ValueError("Squash is only available for 'x', 'y', or 'z'")


    # OpenFOAM has a bug when saving VTK slices using the `type cuttingPlane` option in the dictionary.
    # The bug results in a VTK where the coordinate points are no longer contained in a single plane.
    # For the OpenFOAM-buggy VTK slice, a pattern does seem to exist. It is more clearly explained with
    # an example: Consider a slice at y=4000 and the domain extents are from 0 to 6000 in y. The points
    # are saved at 3 distinct planes: y=4000, y=6000, and another just under 6000, say 5998. The extra
    # points that are not where they should be, are at/near the domain boundary _closest_ to the slice
    # plane location. If the slice was, for example, at y=1000, then the boundary at y=0 will have the 
    # extra points. The only pattern that was identifiable is that 3 unique planes are created and the
    # wrong ones are side by side. The idea followed here is that we identify the "isolated" plane and
    # use it to give the coordinate of the actual sampled location. We recover the correct plane with
    # the commands below.
    # None of this is crucial to this function as is, since it is not returning any information about
    # the plane the VTK is at, but rather only using that information for interpolation purposes. This
    # is left here as a note for future extension of the function.
    if squash in ['x','y','z']:
        # For convenience, we use x and y to refer the dimension the slice varies
        [xminbox, yminbox] = np.min(coords, 0)[[dir1, dir2]]
        [xmaxbox, ymaxbox] = np.max(coords, 0)[[dir1, dir2]]

        uniqueCoords = np.unique(coords[:,dirconst])
        if np.argmax(np.diff(uniqueCoords, prepend=uniqueCoords[0])) == 1:
            zlevel = uniqueCoords[0]
        else:
            zlevel = uniqueCoords[-1]
        zminbox = zmaxbox = zlevel

        # Squash the data into the requested plane
        coords[:,dirconst] = zlevel

    else:
        [xminbox, yminbox, zminbox] = np.min(coords, 0)[[dir1, dir2, dirconst]]
        [xmaxbox, ymaxbox, zmaxbox] = np.max(coords, 0)[[dir1, dir2, dirconst]]



    # Get the proper uniform or non-uniform resolution
    if isinstance(res, (int, float)):
        dx = dy = dz = res
    elif isinstance(res, tuple):
        try: dx=res[0]; dy=res[1]; dz=res[2]
        except IndexError: print('Tuple must have 3 components'); raise
    else:
        raise ValueError('Resolution should be given as either a scalar (uniform '\
                         'resolution) or a 3-component tuple (resx, resy, resz).' )


    # Create the desired output grid
    x1d = np.arange(xminbox,xmaxbox+dx,dx)
    y1d = np.arange(yminbox,ymaxbox+dy,dy)
    if squash in ['x','y','z']:
        z1d = np.asarray([zlevel])    
    else:
        z1d = np.arange(zminbox, zmaxbox+dz, dz)
    [x3d,y3d,z3d] = np.meshgrid(x1d, y1d, z1d, indexing='ij')

    n = np.ravel(x3d).shape[0]            
    coords_want = np.zeros(shape=(n,3))    
    coords_want[:,dir1] = np.ravel(x3d)
    coords_want[:,dir2] = np.ravel(y3d)
    coords_want[:,dirconst] = np.ravel(z3d) 

    # Interpolate values to given resolution
    tmp = griddata(coords, data, coords_want, method='nearest') 

    if tmp.size/n == 3:
        # vector
        out = np.reshape(np.vstack((tmp[:,0],tmp[:,1],tmp[:,2])), (3,x1d.size,y1d.size,z1d.size))
    elif tmp.size/n == 1:
        # scalar
        out=np.reshape(tmp,(1,x1d.size,y1d.size,z1d.size))
    elif tmp.size/n == 6:
        # tensor
        out=np.reshape(np.vstack((tmp[:,0],tmp[:,1],tmp[:,2],tmp[:,3],tmp[:,4],tmp[:,5])), (6,x1d.size,y1d.size,z1d.size))
    else:
        # something else
        raise NotImplementedError(f'Found variable with {tmp.size} components. Not sure how to follow. Stopping.')

    return x1d, y1d, z1d, out



def vtk_write_structured_points( f,
        datadict,
        ds=None,dx=None,dy=None,dz=None,
        origin=(0.0,0.0,0.0),
        indexorder='ijk',
        vtk_header='# vtk DataFile Version 2.0',
        vtk_datatype='float',
        vtk_description='really cool data'
    ):
    """Write a VTK dataset with regular topology to file handle 'f'
    written by Eliot Quon (eliot.quon@nrel.gov)

    Inputs are written with x increasing fastest, then y, then z.

    Example: Writing out two vector fields in one VTK file.
    ```
    from windtools.io.vtk import vtk_write_structured_points
    with open('some_data.vtk','wb') as f:
        vtk_write_structured_points(f,
                                    {'vel':np.stack((u,v,w))},
                                    ds=1.0,
                                    indexorder='ijk')
    ```

    Parameters
    ----------
    datadict : dict
        Dictionary with keys as the data names. Data are either scalar
        fields with shape (nx,ny,nz) or vector fields with shape
        (3,nx,ny,nz).
    ds : float, optional
        Default grid spacing; dx,dy,dz may be specified to override
    dx, dy, dz : float, optional
        Specific grid spacings; if ds is not specified, then all three
        must be specified
    origin : list-like, optional
        Origin of the grid
    indexorder: str
        Specify the indexing convention (standard: 'ijk', TTUDD: 'jik')

    @author: ewquon
    """
    # calculate grid spacings if needed
    if ds:
        if not dx: dx = ds
        if not dy: dy = ds
        if not dz: dz = ds
    else:
        assert( dx > 0 and dy > 0 and dz > 0 ) 

    # check data
    nx = ny = nz = None
    datatype = {}
    for name,data in datadict.items():
        dims = data.shape
        if len(dims) == 3:
            datatype[name] = 'scalar'
        elif len(dims) == 4:
            assert dims[0] == 3
            datatype[name] = 'vector'
        else:
            raise ValueError('Unexpected "'+name+'" array shape: '+str(data.shape))
        if nx is None:
            nx = dims[-3]
            ny = dims[-2]
            nz = dims[-1]
        else:
            assert (nx==dims[-3]) and (ny==dims[-2]) and (nz==dims[-1])

    # write header
    if 'b' in f.mode:
        binary = True
        import struct
        if bytes is str:
            # python 2
            def b(s):
                return str(s)
        else:
            # python 3
            def b(s):
                return bytes(s,'utf-8')
        f.write(b(vtk_header+'\n'))
        f.write(b(vtk_description+'\n'))
        f.write(b('BINARY\n'))
        f.write(b('DATASET STRUCTURED_POINTS\n'))

        # write out mesh descriptors
        f.write(b('DIMENSIONS {:d} {:d} {:d}\n'.format(nx,ny,nz)))
        f.write(b('ORIGIN {:f} {:f} {:f}\n'.format(origin[0],origin[1],origin[2])))
        f.write(b('SPACING {:f} {:f} {:f}\n'.format(dx,dy,dz)))

        # write out data
        f.write(b('POINT_DATA {:d}\n'.format(nx*ny*nz)))

    else:
        binary = False
        f.write(vtk_header+'\n')
        f.write(vtk_description+'\n')
        f.write('ASCII\n')
        f.write('DATASET STRUCTURED_POINTS\n')

        # write out mesh descriptors
        f.write('DIMENSIONS {:d} {:d} {:d}\n'.format(nx,ny,nz))
        f.write('ORIGIN {:f} {:f} {:f}\n'.format(origin[0],origin[1],origin[2]))
        f.write('SPACING {:f} {:f} {:f}\n'.format(dx,dy,dz))

        # write out data
        f.write('POINT_DATA {:d}\n'.format(nx*ny*nz))

    for name,data in datadict.items():
        outputtype = datatype[name]
        if outputtype=='vector':
            u = data[0,:,:,:]
            v = data[1,:,:,:]
            w = data[2,:,:,:]
        elif outputtype=='scalar':
            u = data
        else:
            raise ValueError('Unexpected data type '+outputtype)

        name = name.replace(' ','_')

        mapping = { 'i': range(nx), 'j': range(ny), 'k': range(nz) }
        ijkranges = [ mapping[ijk] for ijk in indexorder ]

        if outputtype=='vector':
            if binary:
                f.write(b('{:s}S {:s} {:s}\n'.format(outputtype.upper(),name,vtk_datatype)))
                for k in ijkranges[2]:
                    for j in ijkranges[1]:
                        for i in ijkranges[0]:
                            f.write(struct.pack('>fff', u[i,j,k], v[i,j,k], w[i,j,k])) # big endian
            else: #ascii
                f.write('{:s}S {:s} {:s}\n'.format(outputtype.upper(),name,vtk_datatype))
                for k in ijkranges[2]:
                    for j in ijkranges[1]:
                        for i in ijkranges[0]:
                            f.write(' {:f} {:f} {:f}\n'.format(u[i,j,k], v[i,j,k], w[i,j,k]))
        elif outputtype=='scalar':
            if binary:
                f.write(b('{:s}S {:s} {:s}\n'.format(outputtype.upper(),name,vtk_datatype)))
                f.write(b('LOOKUP_TABLE default\n'))
                for k in ijkranges[2]:
                    for j in ijkranges[1]:
                        for i in ijkranges[0]:
                            #f.write(struct.pack('f',u[j,i,k])) # native endianness
                            f.write(struct.pack('>f',u[i,j,k])) # big endian
            else:
                f.write('{:s}S {:s} {:s}\n'.format(outputtype.upper(),name,vtk_datatype))
                f.write('LOOKUP_TABLE default\n')
                for k in ijkranges[2]:
                    for j in ijkranges[1]:
                        for i in ijkranges[0]:
                            f.write(' {:f}\n'.format(u[i,j,k]))

