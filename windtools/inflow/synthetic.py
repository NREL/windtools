# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import sys,os
import time
import numpy as np

from .general import InflowPlane
from windtools.io.binary import BinaryFile


class TurbSim(InflowPlane):

    def __init__(self, fname=None, Umean=None, verbose=False, **kwargs):
        """Processes binary full-field time series output from TurbSim.

        Tested with TurbSim v2.00.05c-bjj, 25-Feb-2016
        Tested with pyTurbsim, 10-07-2017
        """
        super(self.__class__,self).__init__(verbose,**kwargs)
        self.Umean = Umean

        if fname is not None:
            self.read_field(fname)


    def read_field(self,fname):
        if not fname.endswith('.bts'):
            fname = fname + '.bts'
        self._readBTS(fname)
        self.have_field = True

    def _readBTS(self,fname):
        """ Process AeroDyn full-field files. Fluctuating velocities and
        coordinates (y & z) are calculated.

        V.shape = (3,NY,NZ,N)  # N: number of time steps
        """
        with BinaryFile(fname) as f:
            #
            # read header info
            #
            if self.verbose: print('Reading header information from',fname)

            ID = f.read_int2()
            assert( ID==7 or ID==8 )
            if ID==7: filetype = 'non-periodic'
            elif ID==8: filetype = 'periodic'
            else: filetype = 'UNKNOWN'
            if self.verbose:
                print('  id= {:d} ({:s})'.format(ID,filetype))

            # - read resolution settings
            self.NZ = f.read_int4()
            self.NY = f.read_int4()
            self.Ntower = f.read_int4()
            if self.verbose:
                print('  NumGrid_Z,_Y=',self.NZ,self.NY)
                print('  ntower=',self.Ntower)
            self.N = f.read_int4()
            self.dz = f.read_float(dtype=self.realtype)
            self.dy = f.read_float(dtype=self.realtype)
            self.dt = f.read_float(dtype=self.realtype)
            self.period  = self.realtype(self.N * self.dt)
            self.Nsize = 3*self.NY*self.NZ*self.N
            if self.verbose:
                print('  nt=',self.N)
                print('  (problem size: {:d} points)'.format(self.Nsize))
                print('  dz,dy=',self.dz,self.dy)
                print('  TimeStep=',self.dt)
                print('  Period=',self.period)

            # - read reference values
            self.uhub = f.read_float(dtype=self.realtype)
            self.zhub = f.read_float(dtype=self.realtype) # NOT USED
            self.zbot = f.read_float(dtype=self.realtype)
            if self.Umean is None:
                self.Umean = self.uhub
                if self.verbose:
                    print('  Umean = uhub =',self.Umean,
                          '(for calculating fluctuations)')
            else: # user-specified Umean
                if self.verbose:
                    print('  Umean =',self.Umean,
                          '(for calculating fluctuations)')
                    print('  uhub=',self.uhub,' (NOT USED)')
            if self.verbose:
                print('  HubHt=',self.zhub,' (NOT USED)')
                print('  Zbottom=',self.zbot)

            # - read scaling factors
            self.Vslope = np.zeros(3,dtype=self.realtype)
            self.Vintercept = np.zeros(3,dtype=self.realtype)
            for i in range(3):
                self.Vslope[i] = f.read_float(dtype=self.realtype)
                self.Vintercept[i] = f.read_float(dtype=self.realtype)
            if self.verbose:
                # output is float64 precision by default...
                print('  Vslope=',self.Vslope)
                print('  Vintercept=',self.Vintercept)

            # - read turbsim info string
            nchar = f.read_int4()
            version = f.read(N=nchar)
            if self.verbose: print(version)

            #
            # read normalized data
            #
            # note: need to specify Fortran-order to properly read data using np.nditer
            t0 = time.process_time()
            if self.verbose: print('Reading normalized grid data')

            self.U = np.zeros((3,self.NY,self.NZ,self.N),order='F',dtype=self.realtype)
            self.T = np.zeros((self.N,self.NY,self.NZ))
            if self.verbose:
                print('  U size :',self.U.nbytes/1024.**2,'MB')

            for val in np.nditer(self.U, op_flags=['writeonly']):
                val[...] = f.read_int2()
            self.U = self.U.swapaxes(3,2).swapaxes(2,1) # new shape: (3,self.N,self.NY,self.NZ)

            if self.Ntower > 0:
                if self.verbose:
                    print('Reading normalized tower data')
                self.Utow = np.zeros((3,self.Ntower,self.N),
                                     order='F',dtype=self.realtype)
                if self.verbose:
                    print('  Utow size :',self.Utow.nbytes/1024.**2,'MB')
                for val in np.nditer(self.Utow, op_flags=['writeonly']):
                    val[...] = f.read_int2()

            if self.verbose:
                print('  Read velocitiy fields in',time.process_time()-t0,'s')
                            
            #
            # calculate dimensional velocity
            #
            if self.verbose:
                print('Calculating velocities from normalized data')
            for i in range(3):
                self.U[i,:,:,:] -= self.Vintercept[i]
                self.U[i,:,:,:] /= self.Vslope[i]
                if self.Ntower > 0:
                    self.Utow[i,:,:] -= self.Vintercept[i]
                    self.Utow[i,:,:] /= self.Vslope[i]
            self.U[0,:,:,:] -= self.Umean # uniform inflow w/ no shear assumed

            print('  u min/max [',np.min(self.U[0,:,:,:]),
                                  np.max(self.U[0,:,:,:]),']')
            print('  v min/max [',np.min(self.U[1,:,:,:]),
                                  np.max(self.U[1,:,:,:]),']')
            print('  w min/max [',np.min(self.U[2,:,:,:]),
                                  np.max(self.U[2,:,:,:]),']')

            self.scaling = np.ones((3,self.NZ))

            #
            # calculate coordinates
            #
            if self.verbose:
                print('Calculating coordinates')
            #self.y = -0.5*(self.NY-1)*self.dy + np.arange(self.NY,dtype=self.realtype)*self.dy
            self.y =             np.arange(self.NY,dtype=self.realtype)*self.dy
            self.z = self.zbot + np.arange(self.NZ,dtype=self.realtype)*self.dz
            #self.ztow = self.zbot - np.arange(self.NZ,dtype=self.realtype)*self.dz #--NOT USED

            self.t = np.arange(self.N,dtype=self.realtype)*self.dt
            if self.verbose:
                print('Read times [',self.t[0],self.t[1],'...',self.t[-1],']')


class GaborKS(InflowPlane):

    def __init__(self, prefix=None,
            tidx=0,
            dt=None, Umean=None,
            potentialTemperature=None,
            verbose=True,
            **kwargs):
        """Processes binary output from Gabor KS.
        """
        super(self.__class__,self).__init__(verbose,**kwargs)
        
        fieldnames = ['uVel','vVel','wVel']
        self.Ncomp = 3
        if potentialTemperature is not None:
            self.Ncomp += 1
            print('Note: Potential temperature is not currently handled!')
            fieldnames.append('potT')

        self.fnames = [ '{}_{}_t{:06d}.out'.format(prefix,fieldvar,tidx) for fieldvar in fieldnames ]
        self.infofile = '{}_info_t{:06d}.out'.format(prefix,tidx)
        self.Umean = Umean
        self.dt = dt

        self.read_info(self.infofile)

        if self.dt is None and self.Umean is None:
            self.dt = 1.0
            self.Umean = self.dx
        elif self.Umean is None:
            self.Umean = self.dx / self.dt
            print('Specified dt =',self.dt)
            print('Calculated Umean =',self.Umean)
        elif self.dt is None:
            self.dt = self.dx / self.Umean
            print('Specified Umean =',self.Umean)
            print('Calculated dt =',self.dt)
        else:
            if self.verbose:
                print('Specified Umean, dt =',self.Umean,self.dt)

        self.t = np.arange(self.NX)*self.dt
        self.y = np.arange(self.NY)*self.dy
        self.z = np.arange(self.NZ)*self.dz
        if self.verbose:
            print('t range:',[np.min(self.t),np.max(self.t)])
            print('y range:',[np.min(self.y),np.max(self.y)])
            print('z range:',[np.min(self.z),np.max(self.z)])

        if self.fnames is not None:
            self.read_field(self.fnames)


    def read_info(self,fname):
        info = np.genfromtxt(fname, dtype=None)
        self.t0 = info[0]
        self.NX = int(info[1])
        self.NY = int(info[2])
        self.NZ = int(info[3])
        self.Lx = info[4]
        self.Ly = info[5]
        self.Lz = info[6]
        self.N = self.NX # time steps equal to x planes
        self.dx = self.Lx/self.NX
        self.dy = self.Ly/self.NY
        self.dz = self.Lz/self.NZ

        self.xG,self.yG,self.zG = np.meshgrid(
                np.linspace(0,self.Lx-self.dx,self.NX),
                np.linspace(0,self.Ly-self.dy,self.NY),
                np.linspace(self.dz/2,self.Lz-(self.dz/2),self.NZ),
                indexing='ij')

        print('Read info file',fname)
        if self.verbose:
            print('  domain dimensions:',[self.NX,self.NY,self.NZ])
            print('  domain extents:',[self.Lx,self.Ly,self.Lz],'m')


    def read_field(self,fnames):
        self.U = np.zeros((self.Ncomp,self.NX,self.NY,self.NZ))
        self.T = np.zeros((self.NX,self.NY,self.NZ))
        self.scaling = np.ones((3,self.NZ))

        for icomp,fname in enumerate(self.fnames):
            tmpdata = np.fromfile(fname,dtype=np.dtype(np.float64),count=-1)
            self.U[icomp,:,:,:] = tmpdata.reshape((self.NX,self.NY,self.NZ),order='F')

        self.have_field = True


