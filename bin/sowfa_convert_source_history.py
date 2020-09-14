#!/usr/bin/env python

# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""
Convert precursor sourceHistory into OpenFOAM dictionaries to include in
constant/ABLProperties as _given_ momentum and/or temperature source
terms.

USAGE: sowfa_convert_source_history.py [/path/to/postProcessing/sourceHistory]

where the default path is ./postProcessing/sourceHistory.

Upon completion, the script will write out:
- constant/givenSourceU
- constant/givenSourceT

"""
import sys, os
import numpy as np
from windtools.SOWFA6.postProcessing.sourceHistory import SourceHistory

try:
    srchistpath = sys.argv[1]
except IndexError:
    srchistpath = 'postProcessing/sourceHistory'
if not os.path.isdir(srchistpath):
    print(srchistpath,'not found; please specify a valid path')

sourceU = SourceHistory(srchistpath,varList='Momentum')
sourceT = SourceHistory(srchistpath,varList='Temperature')

outpath = 'constant'
try:
    open(os.path.join(outpath,'givenSourceU'),'w')
except IOError:
    outpath = '.'

sourceUpath = os.path.join(outpath,'givenSourceU')
with open(sourceUpath,'w') as f:
    fmt = '    %g'
    f.write('sourceHeightsMomentum\n')
    f.write('(\n')
    np.savetxt(f, sourceU.hLevelsCell, fmt=fmt)
    f.write(');\n\n')

    fmt = '    (' + ' '.join((sourceU.N+1)*['%g']) + ')'
    for i,comp in enumerate(['X','Y','Z']):
        srcdata = np.hstack((sourceU.t[:,np.newaxis], sourceU.Momentum[:,:,i]))
        f.write(f'sourceTableMomentum{comp}\n')
        f.write('(\n')
        np.savetxt(f, srcdata, fmt=fmt)
        f.write(');\n\n')
print('Wrote',sourceUpath)

sourceTpath = os.path.join(outpath,'givenSourceT')
with open(sourceTpath,'w') as f:
    fmt = '    %g'
    f.write('sourceHeightsTemperature\n')
    f.write('(\n')
    np.savetxt(f, sourceT.hLevelsCell, fmt=fmt)
    f.write(');\n\n')

    fmt = '    (' + ' '.join((sourceT.N+1)*['%g']) + ')'
    srcdata = np.hstack((sourceT.t[:,np.newaxis], sourceT.Temperature))
    f.write(f'sourceTableTemperature\n')
    f.write('(\n')
    np.savetxt(f, srcdata, fmt=fmt)
    f.write(');\n\n')
print('Wrote',sourceTpath)

