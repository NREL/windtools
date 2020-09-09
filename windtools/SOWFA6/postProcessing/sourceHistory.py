# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""
For processing SOWFA-6 driving force data in postProcessing/SourceHistory
Based on averaging.py written by Eliot Quon

written by Dries Allaerts (dries.allaerts@nrel.gov)

The class can handle both height-dependent and constant source files

Sample usage:

    from windtools.SOWFA6.postProcessing.sourceHistory import SourceHistory

    # read all time directories in current working directory or in subdirectory called 'SourceHistory'
    srcData = SourceHistory()

    # read '0' and '1000' in current working directory
    srcData = SourceHistory( 0, 1000 )

    # read all time directories in specified directory
    srcData = SourceHistory('caseX/postProcessing/SourceHistory')

    # read specified time directories
    srcData = SourceHistory('caseX/postProcessing/SourceHistory/0',
                        'caseX/postProcessing/SourceHistory/1000')
"""
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from .reader import Reader


class SourceHistory(Reader):

    def __init__(self,dpath=None,**kwargs):
        super().__init__(dpath,includeDt=True,**kwargs)


    def _processdirs(self,
                     tdirList,
                     varList=['Momentum','Temperature'],
                     trimOverlap=True
                     ):
        #Redefine _processdirs so that default
        #argument for varList can be specified
        super()._processdirs(tdirList,varList,trimOverlap)
        

    def _read_data(self,dpath,fname):
        fpath = dpath + os.sep + 'Source' + fname + 'History'
        if fname.startswith('Error'):
            fpath = dpath + os.sep + fname + 'History'


        with open(fpath) as f:
            try:
                self._read_source_heights(f)
            except IOError:
                print('unable to read '+fpath)
            else:
                array = self._read_source_data(f)
        return array

    def _read_source_heights(self,f):
        line = f.readline().split()

        if line[0].startswith('Time'):
            self.hLevelsCell = [0.0]
        elif line[0].startswith('Heights'):
            self.hLevelsCell = [ float(val) for val in line[2:] ]
            f.readline()
        else:
            print('Error: Expected first line to start with "Time" or "Heights", but instead read',line[0])
            return

        if (len(self._processed) > 0): # assert that all fields have same number of heights
            assert (self.N == len(self.hLevelsCell)), \
                    'Error: Various source fields do not have the same number of heights, set varList to read separately'
        else: # first field: set number of heights in self.N
            self.N = len(self.hLevelsCell)
        self.hLevelsCell = np.array(self.hLevelsCell)
        return

        
    def _read_source_data(self,f):
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

"""end of class SourceHistory"""
