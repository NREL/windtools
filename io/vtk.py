# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""
Functions to deal with simple VTK I/O.
"""
import struct

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

