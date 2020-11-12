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

import windtools.SOWFA6.constant.boundaryData as bc
from windtools.io.vtk import vtk_write_structured_points


class InflowPlane(object):
    """This is the base class for all inflows. User should create an
    instance of one of the derived classes in the synthetic module.
    """

    realtype = np.float32

    def __init__(self, verbose=False):
        """Base initializer for inflow data, which should be overridden.
        Defaults are set here.

        After initialization, the following variables should be set:
        * Dimensions: NY, NZ (horizontal, vertical)
        * Number of time snapshots: N
        * Spacings/step size: dt or dx, dy, dz
        * Rectilinear grid: y, z
        * Sampling times or streamwisei coordinate: t or x
        * Velocity field: U (with shape==(3,Ntimes,NY,NZ))
        * Potential temperature field: T (with shape==(Ntimes,NY,NZ))
        * Scaling function: scaling (shape==(3,NZ))

        Optionally, the following parameters may be set:
        * Reference velocity: Umean
        """
        self.verbose = verbose
        self.Umean = None # reference velocity
        self.have_field = False # True after the velocity field has been read

#        self.needUpdateMean = False # set to true update the mean inflow at every time step. 
#        self.timeseries = None # used if needUpdateMean is True
#        self.Useries = None # used if needUpdateMean is True
#        self.Tseries = None # used if needUpdateMean is True

        self.mean_flow_read = False
        self.variances_read = False

        # inflow plane coordinates
        self.y = None
        self.z = None

        # set by calcVariance
        self.uu_mean = None
        self.vv_mean = None
        self.ww_mean = None

        # constant profiles, set by readAllProfiles or readVarianceProfile)
        self.z_profile = None
        self.uu_profile = None
        self.vv_profile = None
        self.ww_profile = None


    def read_field(self):
        print('This is a function stub; no inflow data were read.')


    def calcVariance(self,output=None):
        """Calculate variance of the fluctuating velocities. The square
        root (i.e., the standard deviation) should match the output in
        PREFIX.sum.
        """
        self.uu = self.U[0,:,:,:]**2
        self.vv = self.U[1,:,:,:]**2
        self.ww = self.U[2,:,:,:]**2
        self.uu_tavg = np.mean(self.uu,0) # time averages
        self.vv_tavg = np.mean(self.vv,0)
        self.ww_tavg = np.mean(self.ww,0)
        self.uu_mean = np.mean( self.uu_tavg ) # space/time average
        self.vv_mean = np.mean( self.vv_tavg )
        self.ww_mean = np.mean( self.ww_tavg )

        print('Spatial average of <u\'u\'>, <v\'v\'>, <w\'w\'> :',self.uu_mean,self.vv_mean,self.ww_mean)

        if output is not None:
            with open(output,'w') as f:
                f.write('Spatial average of <u\'u\'>, <v\'v\'>, <w\'w\'> : {} {} {}\n'.format(self.uu_mean,self.vv_mean,self.ww_mean))
                f.write('\n   Height   Standard deviation at grid points for the u component:\n')
                for i,zi in enumerate(self.z):
                        f.write('z= {:.1f} : {}\n'.format(zi,np.sqrt(self.uu_tavg[:,i])))
                f.write('\n   Height   Standard deviation at grid points for the v component:\n')
                for i,zi in enumerate(self.z):
                        f.write('z= {:.1f} : {}\n'.format(zi,np.sqrt(self.vv_tavg[:,i])))
                f.write('\n   Height   Standard deviation at grid points for the w component:\n')
                for i,zi in enumerate(self.z):
                        f.write('z= {:.1f} : {}\n'.format(zi,np.sqrt(self.ww_tavg[:,i])))
            print('Wrote out',output)


    #
    # Domain manipulation functions
    #

    def tileY(self,ntiles,mirror=False):
        """Duplicate field in lateral direction
        'ntiles' is the final number of panels including the original

        Set 'mirror' to True to flip every other tile
        """
        ntiles = int(ntiles)
        print('Creating',ntiles,'horizontal tiles')
        print('  before:',self.U.shape)
        if mirror:
            # [0 1 2] --> [0 1 2 1 0 1 2 .. ]
            NYnew = (self.NY-1)*ntiles + 1
            Unew = np.zeros((3,self.N,NYnew,self.NZ))
            Tnew = np.zeros((  self.N,NYnew,self.NZ))
            Unew[:,:,:self.NY,:] = self.U[:,:,:self.NY,:]
            Tnew[  :,:self.NY,:] = self.T[  :,:self.NY,:]
            delta = self.NY - 1
            flipped = True
            for i in range(1,ntiles):
                if flipped:
                    Unew[:,:,i*delta+1:(i+1)*delta+1,:] = self.U[:,:,delta-1::-1,:]
                    Tnew[  :,i*delta+1:(i+1)*delta+1,:] = self.T[  :,delta-1::-1,:]
                else:
                    Unew[:,:,i*delta+1:(i+1)*delta+1,:] = self.U[:,:,1:,:]
                    Tnew[  :,i*delta+1:(i+1)*delta+1,:] = self.T[  :,1:,:]
                flipped = not flipped
            self.U = Unew
            self.T = Tnew
        else:
            # [0 1 2] --> [0 1 0 1 .. 0 1 2]
            self.U = np.tile(self.U[:,:,:-1,:],(1,1,ntiles,1))
            self.T = np.tile(self.T[  :,:-1,:],(  1,ntiles,1))
            Uplane0 = np.zeros((3,self.N,1,self.NZ))
            Tplane0 = np.zeros((  self.N,1,self.NZ))
            Uplane0[:,:,0,:] = self.U[:,:,-1,:]
            Tplane0[  :,0,:] = self.T[  :,-1,:]
            self.U = np.concatenate((self.U,Uplane0),axis=1)
            self.T = np.concatenate((self.T,Tplane0),axis=1)
        print('  after :',self.U.shape)

        self.NY = NYnew
        assert( self.U.shape == (3,self.N,self.NY,self.NZ) )
        self.y = np.arange(self.NY,dtype=self.realtype)*self.dy


    def resizeY(self,yMin=None,yMax=None,dryrun=False):
        """Resize inflow domain to fit LES boundary and update NY.
        Min(y) will be shifted to coincide with yMin.
        """
        if yMin is None:
            yMin = self.y[0]
        if yMax is None:
            yMax = self.y[-1]
        Ly_specified = yMax - yMin
        Ly = self.y[-1] - self.y[0]
        if Ly_specified > Ly:
            print('Specified y range', (yMin,yMax),
                    'greater than', (self.y[0],self.y[-1]))
            return

        if dryrun: sys.stdout.write('(DRY RUN) ')
        print('Resizing fluctuations field in y-dir from [',
                self.y[0],self.y[-1],'] to [',yMin,yMax,']')
        print('  before:',self.U.shape)
        
        newNY = int(np.ceil(Ly_specified/Ly * self.NY))
        Unew = self.U[:,:,:newNY,:]
        Tnew = self.T[  :,:newNY,:]
        print('  after:',Unew.shape)
        if not dryrun:
            self.U = Unew
            self.T = Tnew
            self.NY = newNY

        ynew = yMin + np.arange(newNY,dtype=self.realtype)*self.dy
        if not dryrun:
            print('Updating y coordinates')
            self.y = ynew
        else:
            print('(DRY RUN) y coordinates:',ynew)


    def resizeZ(self,zMin=None,zMax=None,shrink=False,dryrun=False):
        """Set/extend inflow domain to fit LES boundary and update NZ.
        Values between zMin and min(z) will be duplicated from
        V[:3,y,z=min(z),t], whereas values between max(z) and zMax will
        be set to zero.

        By default, this function will not resize inflow plane to a
        smaller domain; to override this, set shrink to True.
        """
        if zMin is None:
            zMin = self.z[0]
        if zMax is None:
            zMax = self.z[-1]
        if not shrink:
            if zMin > self.z[0]:
                print('zMin not changed from',self.z[0],'to',zMin)
                return
            if zMax < self.z[-1]:
                print('zMax not changed from',self.z[-1],'to',zMax)
                return

        self.zbot = zMin

        imin = int((zMin-self.z[0])/self.dz)
        imax = int(np.ceil((zMax-self.z[0])/self.dz))
        zMin = imin*self.dz + self.z[0]
        zMax = imax*self.dz + self.z[0]
        ioff = int((self.z[0]-zMin)/self.dz)
        if dryrun: sys.stdout.write('(DRY RUN) ')
        print('Resizing fluctuations field in z-dir from [',
                self.z[0],self.z[-1],'] to [',zMin,zMax,']')
        print('  before:',self.U.shape)
        
        newNZ = imax-imin+1
        Unew = np.zeros((3,self.N,self.NY,newNZ))
        Tnew = np.zeros((  self.N,self.NY,newNZ))
        for iz in range(ioff):
            Unew[:,:,:,iz] = self.U[:,:,:,0]
            Tnew[  :,:,iz] = self.T[  :,:,0]
        if not shrink:
            Unew[:,:,:,ioff:ioff+self.NZ] = self.U
            Tnew[  :,:,ioff:ioff+self.NZ] = self.T
        else:
            iupper = np.min((ioff+self.NZ, newNZ))
            Unew[:,:,:,ioff:iupper] = self.U[:,:,:,:iupper-ioff]
            Tnew[  :,:,ioff:iupper] = self.T[  :,:,:iupper-ioff]
        print('  after:',Unew.shape)
        if not dryrun:
            self.U = Unew
            self.T = Tnew
            self.NZ = newNZ

        znew = self.zbot + np.arange(newNZ,dtype=self.realtype)*self.dz
        if not dryrun:
            print('Updating z coordinates')
            self.z = znew
        else:
            print('(DRY RUN) z coordinates:',znew)

        if not dryrun:
            print('Resetting scaling function')
            self.scaling = np.ones((3,newNZ))


    #
    # Boundary output
    #

    def write_sowfa_mapped_BC(self,
                              outputdir='boundaryData',
                              time_varying_input=None,
                              ref_height=None,
                              bcname='west',
                              xinlet=0.0,
                              tstart=0.0,
                              periodic=False):
        """For use with OpenFOAM's timeVaryingMappedFixedValue boundary
        condition.  This will create a points file and time directories
        in 'outputdir', which should be placed in
            constant/boundaryData/<patchname>.

        time_varying_input should be a dictionary of (NT, NY, NZ, 3)
        arrays which shoud be aligned with the loaded data in terms of
        (dy, dz, dt, and NT)
        """
        dpath = os.path.join(outputdir, bcname)
        if not os.path.isdir(dpath):
            print('Creating output dir :',dpath)
            os.makedirs(dpath)

        if ref_height is not None: assert(self.z is not None)

        # TODO: check time-varying input
        assert(time_varying_input is not None)
        Uinput = time_varying_input['U']
        Tinput = time_varying_input['T']
        kinput = time_varying_input['k']
        NT, NY, NZ, _ = Uinput.shape
        u = np.zeros((NY,NZ)) # working array
        v = np.zeros((NY,NZ)) # working array
        w = np.zeros((NY,NZ)) # working array
        T = np.zeros((NY,NZ)) # working array

        # write points
        fname = os.path.join(dpath,'points')
        print('Writing',fname)
        with open(fname,'w') as f:
            f.write(bc.pointsheader.format(patchName=bcname,N=NY*NZ))
            for k in range(NZ):
                for j in range(NY):
                    f.write('({:f} {:f} {:f})\n'.format(xinlet,
                                                        self.y[j],
                                                        self.z[k]))
            f.write(')\n')

        # begin time-step loop
        for itime in range(NT):
            curtime = self.realtype(tstart + (itime+1)*self.dt)
            tname = '{:f}'.format(curtime).rstrip('0').rstrip('.')

            prefix = os.path.join(dpath,tname)
            if not os.path.isdir(prefix):
                os.makedirs(prefix)

            # get fluctuations at current time
            if periodic:
                itime0 = np.mod(itime, self.N)
            else:
                itime0 = itime
            #u[:,:] = self.U[0,itime0,:NY,:NZ] # self.U.shape==(3, NT, NY, NZ)
            #v[:,:] = self.U[1,itime0,:NY,:NZ] # self.U.shape==(3, NT, NY, NZ)
            utmp = self.U[0,itime0,:NY,:NZ].copy()
            vtmp = self.U[1,itime0,:NY,:NZ].copy()
            w[:,:] = self.U[2,itime0,:NY,:NZ] # self.U.shape==(3, NT, NY, NZ)
            T[:,:] = self.T[itime0,:NY,:NZ] # self.T.shape==(NT, NY, NZ)

            # scale fluctuations
            for iz in range(NZ): # note: u is the original size
                utmp[:,iz] *= self.scaling[0,iz]
                vtmp[:,iz] *= self.scaling[1,iz]
                w[:,iz] *= self.scaling[2,iz]

            # rotate fluctuating field
            # TODO: allow for constant input
            winddir_profile = np.mean(np.arctan2(Uinput[itime,:,:,1],
                                                 Uinput[itime,:,:,0]), axis=0)
            if ref_height is not None:
                mean_winddir = np.interp(ref_height, self.z, winddir_profile)
            else:
                mean_winddir = np.mean(winddir_profile)
            mean_winddir_compass = 270.0 - 180.0/np.pi*mean_winddir
            if mean_winddir_compass < 0:
                mean_winddir_compass += 360.0
            if ref_height is not None:
                print(('Mean wind dir at {:.1f} m is {:.1f} deg,' \
                    + ' rotating by {:.1f} deg').format(ref_height,
                        mean_winddir_compass, mean_winddir*180.0/np.pi))
            else:
                print(('Mean wind dir is {:.1f} deg,' \
                    + ' rotating by {:.1f} deg').format(
                        mean_winddir_compass, mean_winddir*180.0/np.pi))
            u[:,:] = utmp*np.cos(mean_winddir) - vtmp*np.sin(mean_winddir)
            v[:,:] = utmp*np.sin(mean_winddir) + vtmp*np.cos(mean_winddir)
            
            # superimpose inlet snapshot
            u[:,:] += Uinput[itime,:,:,0]
            v[:,:] += Uinput[itime,:,:,1]
            w[:,:] += Uinput[itime,:,:,2]
            T[:,:] += Tinput[itime,:,:]

            # write out U
            fname = os.path.join(prefix,'U')
            print('Writing out',fname)
            bc.write_data(fname,
                          np.stack((u.ravel(order='F'),
                                    v.ravel(order='F'),
                                    w.ravel(order='F'))),
                          patchName=bcname,
                          timeName=tname,
                          avgValue=[0,0,0])

            # write out T
            fname = os.path.join(prefix,'T')
            print('Writing out',fname)
            bc.write_data(fname,
                          T.ravel(order='F'),
                          patchName=bcname,
                          timeName=tname,
                          avgValue=0)

            # write out k
            fname = os.path.join(prefix,'k')
            print('Writing out',fname)
            bc.write_data(fname,
                          kinput[itime,:,:].ravel(order='F'),
                          patchName=bcname,
                          timeName=tname,
                          avgValue=0)


    #
    # Visualization output
    #

    def writeVTK(self, fname,
            itime=None,
            output_time=None,
            scaled=True,
            stdout='overwrite'):
        """Write out binary VTK file with a single vector field for a
        specified time index or output time.
        """
        if output_time:
            itime = int(output_time / self.dt)
        if itime is None:
            print('Need to specify itime or output_time')
            return
        if stdout=='overwrite':
            sys.stdout.write('\rWriting time step {:d} :  t= {:f}'.format(
                itime,self.t[itime]))
        else: #if stdout=='verbose':
            print('Writing out VTK for time step',itime,': t=',self.t[itime])

        # scale fluctuations
        up = np.zeros((1,self.NY,self.NZ)) # constant x plane (3D array for VTK output)
        wp = np.zeros((1,self.NY,self.NZ))
        vp = np.zeros((1,self.NY,self.NZ))
        up[0,:,:] = self.U[0,itime,:,:]
        vp[0,:,:] = self.U[1,itime,:,:]
        wp[0,:,:] = self.U[2,itime,:,:]
        if scaled:
            for iz in range(self.NZ):
                up[0,:,iz] *= self.scaling[0,iz]
                vp[0,:,iz] *= self.scaling[1,iz]
                wp[0,:,iz] *= self.scaling[2,iz]

        # calculate instantaneous velocity
        U = up.copy()
        V = vp.copy()
        W = wp.copy()
        if self.mean_flow_read:
            for iz in range(self.NZ):
                U[0,:,iz] += self.U_inlet[:,iz]
                V[0,:,iz] += self.V_inlet[:,iz]
                W[0,:,iz] += self.W_inlet[:,iz]

        # write out VTK
        vtk_write_structured_points(
            open(fname,'wb'), #binary mode
            {
                "U": np.stack((U,V,W)),
                "u'": np.stack((up,vp,wp)),
            },
            dx=1.0, dy=self.dy, dz=self.dz,
            origin=[0.,self.y[0],self.z[0]],
            indexorder='ijk',
        )


    def writeVTKSeries(self,
            outputdir='.',
            prefix='inflow',
            step=1,
            scaled=True,
            stdout='overwrite'):
        """Driver for writeVTK to output a range of times"""
        if not os.path.isdir(outputdir):
            print('Creating output dir :',outputdir)
            os.makedirs(outputdir)

        for i in range(0,self.N,step):
            fname = os.path.join(outputdir, f'{prefix:s}_{i:06d}.vtk')
            self.writeVTK(fname,itime=i,scaled=scaled,stdout=stdout)
        if stdout=='overwrite': sys.stdout.write('\n')


    def writeVTKBlock(self,
            fname='turbulence_box.vtk',
            outputdir=None,
            step=1,
            scaled=True):
        """Write out a 3D block wherein the x planes are comprised of
        temporal snapshots spaced (Umean * step * dt) apart.

        This invokes Taylor's frozen turbulence assumption.
        """
        if outputdir is None:
            outputdir = '.'
        elif not os.path.isdir(outputdir):
            print('Creating output dir :',outputdir)
            os.makedirs(outputdir)

        fname = os.path.join(outputdir,fname)
        print('Writing VTK block',fname)

        if self.Umean is not None:
            Umean = self.Umean
        else:
            Umean = 1.0

        # scale fluctuations
        Nt = int(self.N / step)
        up = np.zeros((Nt,self.NY,self.NZ))
        vp = np.zeros((Nt,self.NY,self.NZ))
        wp = np.zeros((Nt,self.NY,self.NZ))
        up[:,:,:] = self.U[0,:Nt*step:step,:,:]
        vp[:,:,:] = self.U[1,:Nt*step:step,:,:]
        wp[:,:,:] = self.U[2,:Nt*step:step,:,:]
        if scaled:
            for iz in range(self.NZ):
                up[:,:,iz] *= self.scaling[0,iz]
                vp[:,:,iz] *= self.scaling[1,iz]
                wp[:,:,iz] *= self.scaling[2,iz]

        # write out VTK
        vtk_write_structured_points( open(fname,'wb'), #binary mode
            Nt, self.NY, self.NZ,
            [ up,vp,wp ],
            datatype=['vector'],
            dx=step*Umean*self.dt, dy=self.dy, dz=self.dz,
            dataname=['u\''],
            origin=[0.,self.y[0],self.z[0]],
            indexorder='ijk')

