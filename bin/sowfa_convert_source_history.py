#!/usr/bin/env python
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
    f.write(')\n\n')

    fmt = '    (' + ' '.join((sourceU.N+1)*['%g']) + ')'
    for i,comp in enumerate(['X','Y','Z']):
        srcdata = np.hstack((sourceU.t[:,np.newaxis], sourceU.Momentum[:,:,i]))
        f.write(f'sourceTableMomentum{comp}\n')
        f.write('(\n')
        np.savetxt(f, srcdata, fmt=fmt)
        f.write(')\n\n')
print('Wrote',sourceUpath)

sourceTpath = os.path.join(outpath,'givenSourceT')
with open(sourceTpath,'w') as f:
    fmt = '    %g'
    f.write('sourceHeightsTemperature\n')
    f.write('(\n')
    np.savetxt(f, sourceT.hLevelsCell, fmt=fmt)
    f.write(')\n\n')

    fmt = '    (' + ' '.join((sourceT.N+1)*['%g']) + ')'
    srcdata = np.hstack((sourceT.t[:,np.newaxis], sourceT.Temperature))
    f.write(f'sourceTableTemperature\n')
    f.write('(\n')
    np.savetxt(f, srcdata, fmt=fmt)
    f.write(')\n\n')
print('Wrote',sourceTpath)

