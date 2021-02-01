# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""
General class for processing SOWFA data

written by Dries Allaerts (dries.allaerts@nrel.gov)

"""
from __future__ import print_function
import os
import numpy as np

class Reader(object):
    """Stores a time array (t), and field arrays as attributes. The
    fields have shape:
        (Nt, N [, Nd])
    Nt is the number of time samples, and N is the number of probes/vertical levels/ ...
    Vectors have an additional dimension to denote vector components.
    Symmetric tensors have an additional dimension to denote tensor components (xx, xy, xz, yy, yz, zz).

    Sample usage:

        from SOWFA.postProcessing.reader import Reader

        # read all data
        data = Reader('postProcessing/<postProcessingTypeName>/')

        # read specified fields only
        data = Reader('postProcessing/<postProcessingTypeName>/',varList=['U','T'])

        data.to_csv('data.csv')

    """
    def __init__(self,dpath=None,includeDt=False,**kwargs):
        """'Find and process all time directories in path dpath"""
        self._processed = []
        self.simTimeDirs = [] #output time names
        self.simStartTimes = [] # start or restart simulation times
        self.imax = None # for truncating time series
        self.Nt = 0 #Number of time samples
        self.N  = 0 #Number of probes/vertical levels
        self.t = None
        self.dt = None
        self.includeDt = includeDt #Time step is part of output

        if not dpath:
            dpath = '.'

        if dpath[-1] == os.sep: dpath = dpath[:-1] # strip trailing slash
        
        # find results
        listing = os.listdir(dpath)
        for dirname in listing:
            if not os.path.isdir(dpath+os.sep+dirname): continue
            try:
                startTime = float(dirname)
            except ValueError:
                # dirname is not a number
                pass
            else:
                self.simTimeDirs.append( dpath+os.sep+dirname )
                self.simStartTimes.append( startTime )

        if len(self.simTimeDirs) == 0:
            # no time directories found; perhaps a single time directory
            # was directly specified
            dirname = os.path.split(dpath)[-1]
            try:
                startTime = float(dirname)
            except ValueError:
                # dirname is not a number
                pass
            else:
                self.simTimeDirs.append( dpath )
                self.simStartTimes.append( startTime )

        # sort results
        self.simTimeDirs = [ x[1] for x in sorted(zip(self.simStartTimes,self.simTimeDirs)) ]
        self.simStartTimes.sort()

        # process all output dirs
        if len(self.simTimeDirs) > 0:
            self._processdirs( self.simTimeDirs, **kwargs )
        else:
            print('No time directories found!')
            
        self._trim_series_if_needed()


    def _processdirs(self,
                     tdirList,
                     varList=[],
                     trimOverlap=True
                    ):
        """Reads all files within an output time directory.
        An object attribute corresponding to the output name
        is updated, e.g.:
            ${timeDir}/U is appended to the array self.U
        """
        print('Simulation (re)start times:',self.simStartTimes)

        if isinstance( varList, (str,) ):
            if varList.lower()=='all':
                # special case: read all vars
                outputs = [ fname for fname in os.listdir(tdirList[0])
                                if os.path.isfile(tdirList[0]+os.sep+fname) ]
            else: # specified single var
                outputs = [varList]
        else: #specified list
            outputs = varList
        
        # process all data
        selected = []
        for field in outputs:
            arrays = [ self._read_data( tdir,field ) for tdir in tdirList ]

            # combine into a single array and trim end of time series
            # (because simulations that are still running can have different
            # array lengths)
            try:
                newdata = np.concatenate(arrays)[:self.imax,:]
            except ValueError:
                print('Could not concatenate the following time-height arrays:')
                for tdir,arr in zip(tdirList, arrays):
                    print(' ', tdir, arr.shape)

            # get rid of overlapped data for restarts
            if trimOverlap:
                if len(selected) == 0:
                    # create array mask
                    tpart = [ array[:,0] for array in arrays ]
                    for ipart,tcutoff in enumerate(self.simStartTimes[1:]):
                        selectedpart = np.ones(len(tpart[ipart]),dtype=bool)
                        try:
                            iend = np.nonzero(tpart[ipart] >= tcutoff)[0][0]
                        except IndexError:
                            # clean restart
                            pass
                        else:
                            # previous simulation didn't finish; overlapped data
                            selectedpart[iend:] = False 
                        selected.append(selectedpart)
                    # last / currently running part
                    selected.append(np.ones(len(tpart[-1]),dtype=bool))
                    selected = np.concatenate(selected)[:self.imax]
                    assert(len(selected) == len(newdata[:,0]))
                elif not (len(newdata[:,0]) == len(selected)):
                    # if simulation is still running, subsequent newdata may
                    # be longer
                    self.imax = min(len(selected), len(newdata[:,0]))
                    selected = selected[:self.imax]
                    newdata = newdata[:self.imax,:]
                # select only unique data
                newdata = newdata[selected,:]
            
            if self.includeDt:
                offset = 2
            else:
                offset = 1
            # reshape field into (Nt,Nz[,Nd]) and set as attribute
            # - note: first column of 'newdata' is time
            if newdata.shape[1] == self.N+offset:
                # scalar
                setattr( self, field, newdata[:,offset:] )
            elif newdata.shape[1] == 3*self.N+offset:
                # vector
                setattr( self, field, newdata[:,offset:].reshape((newdata.shape[0],self.N,3),order='C') )
            elif newdata.shape[1] == 6*self.N+offset:
                # symmetric tensor
                setattr( self, field, newdata[:,offset:].reshape((newdata.shape[0],self.N,6),order='C') )
            else:
                raise IndexError('Unrecognized number of values')
            self._processed.append(field)
            print('  read',field)        # set time arrays
            
        self.t = newdata[:,0]
        self.Nt = len(self.t)
        if self.includeDt:
            self.dt = newdata[:,1]

    def _read_data(self,fpath):
        return None

    def _trim_series_if_needed(self,fields_to_check=None):
        """check for inconsistent array lengths and trim if needed"""
        if fields_to_check is None:
            fields_to_check = self._processed
        for field in fields_to_check:
            try:
                getattr(self,field)
            except AttributeError:
                print('Skipping time series length check for unknown field: ',
                      field)
                fields_to_check.remove(field)
        field_lengths = [ getattr(self,field).shape[0] for field in fields_to_check ]
        if np.min(field_lengths) < np.max(field_lengths):
            self.imax = np.min(field_lengths)
            # need to prune arrays
            print('Inconsistent averaging field lengths... is simulation still running?')
            print('  truncated field histories from',np.max(field_lengths),'to',self.imax)
            self.t = self.t[:self.imax]
            if self.dt is not None:
                self.dt = self.dt[:self.imax]
            self.Nt = len(self.t)
            for field in fields_to_check:
                Ndim = len(getattr(self,field).shape)
                if Ndim == 2:
                    # scalar
                    setattr(self, field, getattr(self,field)[:self.imax,:])
                elif Ndim == 3:
                    # vector/tensor
                    setattr(self, field, getattr(self,field)[:self.imax,:,:])
                else:
                    print('Unknown field type ',field)


    def __repr__(self):
        s = 'Times read: {:d} {:s}\n'.format(self.Nt,str(self.t))
        s+= 'Fields read:\n'
        for field in self._processed:
            s+= '  {:s} : {:s}\n'.format(field,
                                         str(getattr(self,field).shape))
        return s


    #============================================================================
    #
    # DATA I/O
    #
    #============================================================================

    def to_csv(self,fname,**kwargs):
        """Write out specified range of times in a pandas dataframe

        kwargs: see Reader.to_pandas()
        """
        df = self.to_pandas(**kwargs)
        print('Dumping dataframe to',fname)
        df.to_csv(fname)


    def to_pandas(self,itime=None,fields=None,dtype=None):
        """Create pandas dataframe for the specified range of times

        Inputs
        ------
        itime: integer, list
            Time indice(s) to write out; if None, all times are output
        fields: list
            Name of field variables to write out; if None, all variables
            that have been processed are written out
        dtype: type
            Single datatype to which to cast all fields
        """
        import pandas as pd
        # output all vars
        if fields is None:
            fields = self._processed
        # select time range
        if itime is None:
            tindices = range(len(self.t))
        else:
            try:
                iter(itime)
            except TypeError:
                # specified single time index
                tindices = [itime]
            else:
                # specified list of indices
                tindices = itime

        # create dataframes for each height (with time as secondary index)
        # - note: old behavior was to loop over time
        # - note: loop over height is much faster when Nt >> Nz
        print('Creating dataframe for',self.t[tindices])
        dflist = []
        for i in range(self.N):
            data = {}
            for var in fields:
                F = getattr(self,var)
                if len(F.shape)==2:
                    # scalar
                    data[var] = F[tindices,i]
                elif F.shape[2]==3:
                    # vector
                    for j,name in enumerate(['x','y','z']):
                        data[var+name] = F[tindices,i,j]
                elif F.shape[2]==6:
                    # symmetric tensor
                    for j,name in enumerate(['xx','xy','xz','yy','yz','zz']):
                        data[var+name] = F[tindices,i,j]
            data['t'] = self.t[tindices]
            df = pd.DataFrame(data=data,dtype=dtype)
            df['z'] = self.hLevelsCell[i]
            dflist.append(df)
        return pd.concat(dflist).sort_values(['t','z']).set_index(['t','z'])


    def to_netcdf(self,fname,fieldDescriptions={},fieldUnits={}):
        print('Dumping data to',fname)
        import netCDF4
        f = netCDF4.Dataset(fname,'w')
        f.createDimension('time',len(self.t))
        f.createDimension('z',len(self.hLevelsCell))

        times = f.createVariable('time', 'float', ('time',))
        times.long_name = 'Time'
        times.units = 's'
        times[:] = self.t

        heights = f.createVariable('z', 'float', ('z',))
        heights.long_name = 'Height above ground level'
        heights.units = 'm'
        heights[:] = self.hLevelsCell

        for var in self._processed:
            F = getattr(self,var)
            if len(F.shape)==2:
                # scalar
                varnames = [var,]
                F = F[:,:,np.newaxis]
            elif F.shape[2]==3:
                # vector
                varnames = [var+name for name in ['x','y','z']]
            elif F.shape[2]==6:
                # symmetric tensor
                varnames = [var+name for name in ['xx','xy','xz','yy','yz','zz']]
            
            for i, varname in enumerate(varnames):
                field = f.createVariable(varname, 'float', ('time','z'))
                try:
                    field.long_name = fieldDescriptions[varname]
                except KeyError:
                    # Use var name as description
                    field.long_name = varname
                try:
                    field.units = fieldUnits[varname]
                except KeyError:
                    # Units unknown
                    pass
                field[:] = F[:,:,i]
        f.close()
