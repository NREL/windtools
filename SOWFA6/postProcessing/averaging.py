# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""
For processing SOWFA-6 planar averages

Based on original SOWFA/postProcessing/averaging.py from github.com/NWTC/datatools

written by
- Dries Allaerts (dries.allaerts@nrel.gov)
- Eliot Quon (eliot.quon@nrel.gov)

Sample usage:

    from windtools.SOWFA6.postProcessing.averaging import PlanarAverages

    # read all time directories in current working directory
    averagingData = PlanarAverages()

    # read '0' and '1000' in current working directory
    averagingData = planarAverages( 0, 1000 )

    # read all time directories in specified directory
    averagingData = PlanarAverages('caseX/postProcessing/averaging')

    # read specified time directories
    averagingData = PlanarAverages('caseX/postProcessing/averaging/0',
                        'caseX/postProcessing/averaging/1000')

"""
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from .reader import Reader

class PlanarAverages(Reader):

    def __init__(self,dpath=None,**kwargs):
        super().__init__(dpath,includeDt=True,**kwargs)


    def _processdirs(self,
                     tdirList,
                     varList=['U','T'],
                     trimOverlap=True
                     ):
        #Redefine _processdirs so that default
        #argument for varList can be specified
        super()._processdirs(tdirList,varList,trimOverlap)
        

    def _read_data(self,dpath,fname):
        fpath = dpath + os.sep + fname
        with open(fpath) as f:
            try:
                self._read_heights(f)
            except IOError:
                print('unable to read '+fpath)
            else:
                array = self._read_field_data(f)
        return array

    def _read_heights(self,f):
        line = f.readline().split()
        assert (line[0] == 'Heights'), \
                'Error: Expected first line to start with "Heights", but instead read'+line[0]

        self.hLevelsCell = [ float(val) for val in line[2:] ]
        f.readline()

        if (len(self._processed) > 0): # assert that all fields have same number of heights
            assert (self.N == len(self.hLevelsCell)), \
                    'Error: Various fields do not have the same number of heights'
        else: # first field: set number of heights in self.N
            self.N = len(self.hLevelsCell)
        self.hLevelsCell = np.array(self.hLevelsCell)

        
    def _read_field_data(self,f):
        out = []
        for line in f:
            line = [ float(val) for val in
                    line.replace('(','').replace(')','').split() ]
            out.append(line)
        return np.array(out)


    #============================================================================
    #
    # DATA I/O
    #
    #============================================================================

    def to_netcdf(self,fname):
        fieldDescriptions = {'T': 'Potential temperature',
                      'Ux': 'U velocity component',
                      'Uy': 'V velocity component',
                      'Uz': 'W velocity component',
                      }
        fieldUnits = {'T': 'K',
                 'Ux': 'm s-1',
                 'Uy': 'm s-1',
                 'Uz': 'm s-1',
                }
        super().to_netcdf(fname,fieldDescriptions,fieldUnits)

"""end of class PlanarAverages"""
