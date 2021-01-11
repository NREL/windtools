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


def read_mesh(fpath,headerlength=8,chunksize=None):
    """Read Ensight mesh file (ascii) into a dataframe"""
    with open(fpath,'r') as f:
        for _ in range(headerlength):
            f.readline()
        N = int(f.readline())
        if chunksize is None:
            mesh = pd.read_csv(f,header=None,nrows=3*N).values
        else:
            mesh = pd.concat(pd.read_csv(f,header=None,nrows=3*N,chunksize=chunksize)).values
    df = pd.DataFrame(data=mesh.reshape((N,3),order='F'), columns=['x','y','z'])
    return df


def read_vector(fpath,mesh,t=0.0,headerlength=4,chunksize=None):
    """Read Ensight data array (ascii) into a dataframe with combined mesh
    information corresponding to the specified time; mesh should be read in by
    the read_mesh() function
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
    df['t'] = t
    uvw = pd.DataFrame(data=vals.reshape((Npts,3),order='F'), columns=['u','v','w'])
    df = pd.concat([df,uvw], axis=1)
    return df.set_index(['t','x','y','z'])
