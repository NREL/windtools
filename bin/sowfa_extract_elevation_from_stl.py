#!/usr/bin/env python

# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import sys, glob
import numpy as np
from scipy.interpolate import griddata
# pip install numpy-stl
from stl import mesh

stlpath = glob.glob('constant/triSurface/*.stl')
interp_method = 'cubic' # linear, nearest, cubic

if (len(sys.argv) < 2):
    sys.exit('USAGE: '+sys.argv[0]+' x,y [x,y] ...')
assert (len(stlpath) == 1), 'Did not find single stl file in constant/triSurface'

print('Reading stl from',stlpath[0])
msh = mesh.Mesh.from_file(stlpath[0])
x = msh.vectors[:,:,0].ravel()
y = msh.vectors[:,:,1].ravel()
z = msh.vectors[:,:,2].ravel()

# construct output points
xout = []
yout = []
for xy in sys.argv[1:]:
    assert (',' in xy)
    xyvals = [ float(val) for val in xy.split(',') ]
    assert (len(xyvals) == 2)
    xout.append(xyvals[0])
    yout.append(xyvals[1])

points = np.stack((x,y), axis=-1)
xi = np.stack((xout,yout), axis=-1)
elev = griddata(points, z, xi, method=interp_method)

for xo,yo,zo in zip(xout,yout,elev):
    print(xo,yo,zo)

