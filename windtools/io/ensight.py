# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import os
import pandas as pd


def read_mesh(fpath,headerlength=8,chunksize=None,read_connectivity=False,verbose=False):
    """Read Ensight mesh file (ascii) into a dataframe
    
    Parameters
    ----------
    headerlength: integer
        Number of header lines to skip
    chunksize: integer or None
        Chunksize parameter for pd.read_csv, can speed up I/O
    read_connectivity: bool
        If True, read in connectivity information and convert points
        from nodes to cell centers
    """
    with open(fpath,'r') as f:
        # header
        for _ in range(headerlength):
            line = f.readline()
            if verbose:
                print(line,end='')
        N = int(f.readline())

        # read points
        if chunksize is None:
            mesh = pd.read_csv(f,header=None,nrows=3*N).values
        else:
            mesh = pd.concat(pd.read_csv(f,header=None,nrows=3*N,chunksize=chunksize)).values
        if verbose:
            print(f'read {N} points')
    mesh = pd.DataFrame(data=mesh.reshape((N,3),order='F'), columns=['x','y','z'])

    if read_connectivity:
        # pd.read_csv may read data in chunks, so reading the next line is not
        # guaranteed to give what we want... reread
        with open(fpath,'r') as f:
            for _ in range(headerlength + 1 + 3*N):
                f.readline()
            element_type = f.readline().strip()
            if not element_type == 'quad4':
                print(f'WARNING: element type "{element_type}" not tested')
            Ncell = int(f.readline())
            if chunksize is None:
                conn = pd.read_csv(f,header=None,nrows=Ncell,
                                   delim_whitespace=True)
            else:
                conn = pd.concat(pd.read_csv(f,header=None,nrows=Ncell,
                                             delim_whitespace=True,
                                             chunksize=chunksize))
        if verbose:
            print(f'read connectivity data for {Ncell} cells')
        # switch to 0-indexing
        assert (conn.values.max() == N)
        conn -= 1
        assert (conn.values.min() == 0)
        # calculate cell centers
        newindices = pd.RangeIndex(Ncell)
        nodes = [mesh.loc[indices].set_index(newindices) for col,indices in conn.iteritems()]
        cellcenters = sum(nodes) / len(nodes)
        mesh = cellcenters

    return mesh


def read_vector(fpath,mesh,t=None,sort=False,headerlength=4,chunksize=None):
    """Read Ensight data array (ascii) into a dataframe with combined mesh
    information corresponding to the specified time; mesh should be read in by
    the read_mesh() function
    
    Parameters
    ----------
    t: float
        Simulation time to associate with this file [s], useful if
        reading many files with the intention of using pd.concat
    sort: bool
        Sort by x,y,z
    headerlength: integer
        Number of header lines to skip
    chunksize: integer or None
        Chunksize parameter for pd.read_csv, can speed up I/O
    """
    Npts = len(mesh)
    with open(fpath,'r') as f:
        for _ in range(headerlength):
            f.readline()
        if chunksize is None:
            vals = pd.read_csv(f,header=None,nrows=3*Npts).values
        else:
            vals = pd.concat(pd.read_csv(f,header=None,nrows=3*Npts,chunksize=chunksize)).values
    df = mesh.copy()
    uvw = pd.DataFrame(data=vals.reshape((Npts,3),order='F'), columns=['u','v','w'])
    df = pd.concat([df,uvw], axis=1)
    if t is not None:
        df['t'] = t
        df = df.set_index(['t','x','y','z'])
    else:
        df = df.set_index(['x','y','z'])
    if sort:
        df = df.sort_index()
    return df
