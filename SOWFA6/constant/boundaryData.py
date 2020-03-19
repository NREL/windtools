#
# Module to handle SOWFA boundary data that belong in
#   casedir/constant/boundaryData
#
# Written by Eliot Quon (eliot.quon@nrel.gov)
#
import os
import numpy as np
import matplotlib.pyplot as plt

from windtools.io.series import TimeSeries

contour_colormap = 'RdBu_r' # more soothing blues and reds

pointsheader = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2.4.x                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      {fmt:s};
    class       vectorField;
    location    "constant/boundaryData/{patchName:s}";
    object      points;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

{N:d}
("""

dataheader = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2.4.x                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      {fmt:s};
    class       {patchType:s}AverageField;
    location    "constant/boundaryData/{patchName:s}/{timeName:s}";
    object      values;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// Average
{avgValue:s}

{N:d}
("""


def write_points(fname,x,y,z,patchName='patch',
                 fmt='%f',order='C'):
    """Write out a points file which should be stored in
        constant/boundaryData/patchName/points
    """
    N = len(x)
    assert(N == len(y) == len(z))
    if len(x.shape) > 1:
        x = x.ravel(order=order)
        y = y.ravel(order=order)
        z = z.ravel(order=order)
        N = len(x)
    dpath = os.path.split(fname)[0]
    if not os.path.isdir(dpath):
        os.makedirs(dpath)
    fmtstr = '({:s} {:s} {:s})'.format(fmt,fmt,fmt)
    np.savetxt(fname,
               np.stack((x,y,z)).T, fmt=fmtstr,
               header=pointsheader.format(patchName=patchName,N=N,fmt='ascii'),
               footer=')',
               comments='')

def write_data(fname,
               data,
               patchName='patch',
               timeName=0,
               avgValue=None):
    """Write out a boundaryData file which should be stored in
        constant/boundarydata/patchName/timeName/fieldName

    Parameters
    ----------
    fname : str
        Output data file name
    data : numpy.ndarray
        Field data to be written out, with shape (3,N) for vectors
        and shape (N) for scalars; 2-D or 3-D data should be flattened
        beforehand.
    patchName : str, optional
        Name of the boundary patch
    timeName : scalar or str, optional
        Name of corresponding time directory
    avgValue : scalar or list-like, optional
        To set avgValue in the data file; probably not used.

    @author: ewquon
    """
    dims = data.shape
    N = dims[-1]
    if len(dims) == 1:
        patchType = 'scalar'
        if avgValue is None:
            avgValueStr = '0'
        else:
            avgValueStr = str(avgValue)
    elif len(dims) == 2:
        patchType = 'vector'
        assert(dims[0] == 3)
        if avgValue is None:
            avgValueStr = '(0 0 0)'
        else:
            avgValueStr = '(' + ' '.join([str(val) for val in list(avgValue)]) + ')'
    else:
        print('ERROR: Unexpected number of dimensions! No data written.')
        return

    dpath = os.path.split(fname)[0]
    if not os.path.isdir(dpath):
        os.makedirs(dpath)

    headerstr = dataheader.format(patchType=patchType,
                                  patchName=patchName,
                                  timeName=str(timeName),
                                  avgValue=avgValueStr,
                                  N=N,
                                  fmt='ascii')
    if patchType == 'vector':
        np.savetxt(fname,
                   data.T, fmt='(%g %g %g)',
                   header=headerstr, footer=')',
                   comments='')
    elif patchType == 'scalar':
        np.savetxt(fname,
                   data.reshape((N,1)), fmt='%g',
                   header=headerstr, footer=')',
                   comments='')


def get_unique_points_from_list(ylist,zlist,NY=None,NZ=None,order='F'):
    """Detects y and z (1-D arrays) from a list of points on a
    structured grid. Makes no assumptions about the point
    ordering
    """
    ylist = np.array(ylist)
    zlist = np.array(zlist)
    N = len(zlist)
    assert(N == len(ylist))
    if (NY is not None) and (NZ is not None):
        # use specified plane dimensions
        assert(NY*NZ == N)
        y = ylist.reshape((NY,NZ))[:,0]
    elif zlist[1]==zlist[0]:
        # y changes faster, F-ordering
        NY = np.nonzero(zlist > zlist[0])[0][0]
        NZ = int(N / NY)
        assert(NY*NZ == N)
        y = ylist[:NY]
        z = zlist.reshape((NY,NZ),order='F')[0,:]
    elif ylist[1]==ylist[0]:
        # z changes faster, C-ordering
        NZ = np.nonzero(ylist > ylist[0])[0][0]
        NY = int(N / NZ)
        assert(NY*NZ == N)
        z = zlist[:NZ]
        y = ylist.reshape((NY,NZ),order='C')[:,0]
    else:
        print('Unrecognized point distribution')
        print('"y" :',len(ylist),ylist)
        print('"z" :',len(zlist),zlist)
        return ylist,zlist,False
    return y,z,True


def read_points(fname,tol=1e-6,return_const=False,**kwargs):
    """Returns a 2D set of points if one of the coordinates is constant
    otherwise returns a 3D set of points. Assumes that the points are on a
    structured grid.
    """
    N = None
    points = None
    iread = 0
    with open(fname,'r') as f:
        while N is None:
            try:
                N = int(f.readline())
            except ValueError: pass
            else:
                points = np.zeros((N,3))
                print('Reading',N,'points from',fname)
        for line in f:
            line = line[:line.find('\\')].strip()
            try:
                points[iread,:] = [ float(val) for val in line[1:-1].split() ]
            except (ValueError, IndexError): pass
            else:
                iread += 1
    assert(iread == N)

   #constX = np.all(points[:,0] == points[0,0])
   #constY = np.all(points[:,1] == points[0,1])
   #constZ = np.all(points[:,2] == points[0,2])
    constX = np.max(points[:,0]) - np.min(points[0,0]) < tol
    constY = np.max(points[:,1]) - np.min(points[0,1]) < tol
    constZ = np.max(points[:,2]) - np.min(points[0,2]) < tol
    print('Constant in x/y/z :',constX,constY,constZ)
    if not (constX or constY):
        print('Warning: boundary is not constant in X or Y?')

    if constX:
        x0 = np.mean(points[:,0])
        ylist = points[:,1]
        zlist = points[:,2]
    elif constY:
        x0 = np.mean(points[:,1])
        ylist = points[:,0]
        zlist = points[:,2]
    elif constZ:
        x0 = np.mean(points[:,2])
        ylist = points[:,0]
        zlist = points[:,1]
    else:
        print('Unexpected boundary orientation, returning full list of points')
        return points

    y,z,is_structured = get_unique_points_from_list(ylist,zlist,**kwargs)
    assert(is_structured)
    if return_const:
        return x0,y,z
    else:
        return y,z


def read_vector_data(fname,Ny=None,Nz=None,order='C',verbose=False):
    """Read vector field data from a structured boundary data patch"""
    N = None
    data = None
    iread = 0
    with open(fname,'r') as f:
        for line in f:
            if N is None:
                try:
                    N = int(line)
                    if (Ny is not None) and (Nz is not None):
                        if not N == Ny*Nz:
                            Ny = None
                            Nz = None
                    data = np.zeros((N,3))
                    if verbose: print('Reading',N,'vectors from',fname)
                except ValueError: pass
            elif not line.strip() in ['','(',')',';'] \
                    and not line.strip().startswith('//'):
                data[iread,:] = [ float(val) for val in line.strip().strip('()').split() ]
                iread += 1
    assert(iread == N)

    if (Ny is not None) and (Nz is not None):
        vectorField = np.zeros((3,Ny,Nz))
        for i in range(3):
            vectorField[i,:,:] = data[:,i].reshape((Ny,Nz),order=order)
    else:
        vectorField = data.T

    return vectorField


def read_scalar_data(fname,Ny=None,Nz=None,order='C',verbose=False):
    """Read scalar field data from a structured boundary data patch"""
    N = None
    data = None
    iread = 0
    with open(fname,'r') as f:
        for line in f:
            if (N is None) or N < 0:
                try:
                    if N is None: 
                        avgval = float(line)
                        N = -1 # skip first scalar, which is the average field value (not used)
                    else:
                        assert(N < 0)
                        N = int(line) # now read the number of points
                        if (Ny is not None) and (Nz is not None):
                            if not N == Ny*Nz:
                                Ny = None
                                Nz = None
                        data = np.zeros(N)
                        if verbose: print('Reading',N,'scalars from',fname)
                except ValueError: pass
            elif not line.strip() in ['','(',')',';'] \
                    and not line.strip().startswith('//'):
                data[iread] = float(line)
                iread += 1
    assert(iread == N)

    if (Ny is not None) and (Nz is not None):
        scalarField = data.reshape((Ny,Nz),order=order)
    else:
        scalarField = data

    return scalarField


class BoundaryData(object):
    """Object to handle boundary data"""

    def __init__(self,
                 bdpath,Ny=None,Nz=None,order='F',
                 fields=['U','T','k'],
                 verbose=True):
        """Process timeVaryingMapped* boundary data located in in
            constant/boundaryData/<name>
        """
        self.dpath = bdpath
        assert(os.path.isdir(bdpath))
        self.name = os.path.split(bdpath)[-1]

        self.ts = TimeSeries(bdpath,dirs=True,verbose=verbose)
        self.t = np.array(self.ts.times)
        self.Ntimes = self.ts.Ntimes

        kwargs = {}
        if (Ny is not None) and (Nz is not None):
            kwargs = dict(Ny=Ny, Nz=Nz)
        self.y, self.z = read_points(os.path.join(bdpath,'points'), **kwargs)
        self.Ny = len(self.y)
        self.Nz = len(self.z)

        #self.yy, self.zz = np.meshgrid(self.y, self.z, indexing='ij')
        dy = np.diff(self.y)
        dz = np.diff(self.z)
        assert(np.all(dy==dy[0])) # uniform spacing assumed
        assert(np.all(dz==dz[0]))
        dy = dy[0]
        dz = dz[0]
        self.yy, self.zz = np.meshgrid(np.arange(self.Ny+1)*dy + self.y[0]-dy/2,
                                       np.arange(self.Nz+1)*dz + self.z[0]-dz/2,
                                       indexing='ij')

        # image left, right, bottom, top (for imshow)
        self.extent = (self.y[0], self.y[-1],
                       self.z[-1], self.z[0]) # note top/bottom flipped

        self.field = {}
        haveU, haveT, havek = False, False, False
        if 'U' in fields:
            haveU = True
            self.field['U'] = np.zeros((self.Ntimes,self.Ny,self.Nz))
            self.field['V'] = np.zeros((self.Ntimes,self.Ny,self.Nz))
            self.field['W'] = np.zeros((self.Ntimes,self.Ny,self.Nz))
        if 'T' in fields:
            haveT = True
            self.field['T'] = np.zeros((self.Ntimes,self.Ny,self.Nz))
        if 'k' in fields:
            havek = True
            self.field['k'] = np.zeros((self.Ntimes,self.Ny,self.Nz))

        for itime, dpath in enumerate(self.ts):
            if verbose:
                print('t={:f} {:s}'.format(self.ts.times[itime],dpath))
            Ufield = read_vector_data(os.path.join(dpath, 'U'),
                                      Ny=self.Ny, Nz=self.Nz)
            self.field['U'][itime,:,:] = Ufield[0,:,:]
            self.field['V'][itime,:,:] = Ufield[1,:,:]
            self.field['W'][itime,:,:] = Ufield[2,:,:]
            Tfield = read_scalar_data(os.path.join(dpath, 'T'),
                                      Ny=self.Ny, Nz=self.Nz)
            self.field['T'][itime,:,:] = Tfield
            kfield = read_scalar_data(os.path.join(dpath, 'k'),
                                      Ny=self.Ny, Nz=self.Nz)
            self.field['k'][itime,:,:] = kfield

    def __getattr__(self,name):
        if name in self.field.keys():
            return self.field[name]
        else:
            raise AttributeError("Boundary data do not include field '{:s}'".format(name))

    
    def to_npz(self,fpath='boundaryData.npz'):
        np.savez_compressed(fpath,**self.field)


    def create(self,name,field):
        assert(np.all(field.shape == self.field['U'].shape))
        self.field[name] = field


    def _plot_patch(self,F,name,time,value_range):
        fig = plt.figure(1,figsize=(8,6))
        itime = np.argmin(np.abs(time - self.t))
        #print(itime,time)
#        F = self.field[field] #[itime,:,:]
        if self._init_patch_plot:
            # reset range controls
            fieldmin = np.min(F)
            fieldmax = np.max(F)
            minval = fieldmin
            maxval = fieldmax
            self._update_vrange_minmax(fieldmin,fieldmax)
            self.vrange.value = (fieldmin,fieldmax)
            self._init_patch_plot = False
        else:
            minval, maxval = value_range
        ax = plt.gca()
        cmesh = ax.pcolormesh(self.yy, self.zz, F[itime,:,:],
                              cmap='RdBu_r',
                              vmin=minval, vmax=maxval)

        ax.set_aspect('equal','box')
        ax.set_title('{:s} (i={:d}: t={:g} s)'.format(name,itime,self.t[itime]))
        cbar = plt.colorbar(cmesh,ax=ax,orientation='horizontal',fraction=0.04)
        cbar.set_label(name)
    
    def _update_vrange_minmax(self,fieldmin,fieldmax):
        if fieldmin < self.vrange.max:
            self.vrange.min = fieldmin
            self.vrange.max = fieldmax
        else:
            self.vrange.max = fieldmax
            self.vrange.min = fieldmin

    def iplot(self,name):
        from ipywidgets import interactive, fixed  #interact, interactive, fixed, interact_manual
        import ipywidgets as widgets
        from IPython.display import display
#        allfields = list(self.field.keys())
        F = self.field[name]
        fieldmin = np.min(F)
        fieldmax = np.max(F)
        timeselector = widgets.FloatSlider(
                            min=self.ts.times[0],
                            max=self.ts.times[-1],
                            step=(self.ts.times[1]-self.ts.times[0]),
                            value=self.ts.times[0]
                        )
        self.vrange = widgets.FloatRangeSlider(
                                min=fieldmin,
                                max=fieldmax,
                                step=(fieldmax-fieldmin)/10,
                                value=(fieldmin,fieldmax)
                            )

        self._init_patch_plot = True
        self.plotwidget = interactive(
                                self._plot_patch,
                                F=fixed(F),
                                name=fixed(name),
                                time=timeselector,
                                value_range=self.vrange,
                            )
        display(self.plotwidget)


class CartesianPatch(object):
    """Object to facilitate outputing boundary patches on a Cartesian
    grid, with boundaries at x=const or y=const.
    """

    def __init__(self,x,y,z,dpath='.',name='patch'):
        """For a Cartesian mesh, the grid is defined by 1-D coordinate
        vectors x,y,z
        """
        self.dpath = dpath
        self.name = name
        # check for constant x/y
        if np.all(x==x[0]):
            self.desc = 'const x={:.1f}'.format(x[0])
            x = [x[0]]
        elif np.all(y==y[0]):
            self.desc = 'const y={:.1f}'.format(y[0])
            y = [y[0]]
        else:
            raise ValueError('x and y not constant, domain is not Cartesian')
        # set up points
        self.x = x
        self.y = y
        self.z = z
        self.Nx = len(x)
        self.Ny = len(y)
        self.Nz = len(z)
        # set up mesh
        self.X, self.Y, self.Z = np.meshgrid(x,y,z,indexing='ij')

    def __repr__(self):
        s = 'Cartesian patch "{:s}" : '.format(self.name)
        s += self.desc
        s +=', size=({:d},{:d},{:d})'.format(self.Nx,self.Ny,self.Nz)
        return s

    def write_points(self,fpath=None):
        """Write out constant/boundaryData/patchName/points file"""
        if fpath is None:
            fpath = os.path.join(self.dpath,self.name,'points')
        write_points(fpath,self.X,self.Y,self.Z,patchName=self.name)
        print('Wrote points to '+fpath)

    def write_profiles(self, t, z,
                       U=None, V=None, W=None, T=None, k=None,
                       time_range=[None,None],
                       verbose=True):
        """Write out constant/boundaryData/patchName/*/{U,T} given a
        set of time-height profiles. Outputs will be interpolated to 
        the patch heights.

        Inputs
        ------
        t : np.ndarray
            Time vector with length Nt
        z : np.ndarray
            Height vector with length Nz
        U : np.ndarray
            Velocity vectors (Nt,Nz,3) or x-velocity component (Nt,Nz)
        V : np.ndarray, optional
            y-velocity component (Nt,Nz)
        W : np.ndarray, optional
            z-velocity component (Nt,Nz)
        T : np.ndarray
            potential temperature (Nt,Nz)
        time_range : tuple, optional
            range of times to write out boundary data
        """
        assert(U is not None)
        assert(T is not None)
        Nt, Nz = U.shape[:2]
        assert(Nt == len(t))
        assert(Nz == len(z))
        if len(U.shape) == 2:
            have_components = True
            assert(np.all(U.shape == (Nt,Nz)) and \
                    np.all(V.shape == (Nt,Nz)) and \
                    np.all(W.shape == (Nt,Nz)) and \
                    np.all(T.shape == (Nt,Nz)))
        elif len(U.shape) == 3:
            have_components = False
            assert(U.shape[2] == 3)
            assert(np.all(T.shape == (Nt,Nz)))
        else:
            raise InputError

        if not have_components:
            V = U[:,:,1]
            W = U[:,:,2]
            U = U[:,:,0]

        if k is not None:
            assert(np.all(k.shape == (Nt,Nz)))

        if time_range[0] is None:
            time_range[0] = 0.0
        if time_range[1] is None:
            time_range[1] = 9e9

        if (not len(self.z) == Nz) or (not np.all(self.z == z)):
            interpolate = True
        else:
            # input z are equal to patch z
            interpolate = False

        Upatch = np.zeros((self.Nx,self.Ny,self.Nz))
        Vpatch = np.zeros((self.Nx,self.Ny,self.Nz))
        Wpatch = np.zeros((self.Nx,self.Ny,self.Nz))
        Tpatch = np.zeros((self.Nx,self.Ny,self.Nz))
        kpatch = np.zeros((self.Nx,self.Ny,self.Nz))
        for it,ti in enumerate(t):
            if (ti < time_range[0]) or (ti > time_range[1]):
                if verbose:
                    print('skipping time '+str(ti))
                continue
            tname = '{:f}'.format(ti).rstrip('0').rstrip('.')

            timepath = os.path.join(self.dpath, self.name, tname)
            if not os.path.isdir(timepath):
                os.makedirs(timepath)

            Upatch[:,:,:] = 0.0
            Vpatch[:,:,:] = 0.0
            Wpatch[:,:,:] = 0.0
            Tpatch[:,:,:] = 0.0
            kpatch[:,:,:] = 0.0
            if interpolate:
                if verbose:
                    print('interpolating data at t = {:s} s'.format(tname)) 
                for iz,zp in enumerate(self.z):
                    i = np.nonzero(z > zp)[0][0]
                    f = (zp - z[i-1]) / (z[i] - z[i-1])
                    Upatch[:,:,iz] = U[it,i-1] + f*(U[it,i]-U[it,i-1])
                    Vpatch[:,:,iz] = V[it,i-1] + f*(V[it,i]-V[it,i-1])
                    Wpatch[:,:,iz] = W[it,i-1] + f*(W[it,i]-W[it,i-1])
                    Tpatch[:,:,iz] = T[it,i-1] + f*(T[it,i]-T[it,i-1])
                    if k is not None:
                        kpatch[:,:,iz] = k[it,i-1] + f*(k[it,i]-k[it,i-1])
            else:
                if verbose:
                    print('mapping data at t = {:s} s'.format(tname)) 
                for iz in range(Nz):
                    Upatch[:,:,iz] = U[it,iz]
                    Vpatch[:,:,iz] = V[it,iz]
                    Wpatch[:,:,iz] = W[it,iz]
                    Tpatch[:,:,iz] = T[it,iz]
                    if k is not None:
                        kpatch[:,:,iz] = k[it,iz]
        
            Upath = os.path.join(timepath,'U')
            Udata = np.stack((Upatch.ravel(), Vpatch.ravel(), Wpatch.ravel()))
            if verbose: print('writing data to {:s}'.format(Upath)) 
            write_data(Upath, Udata, patchName=self.name, timeName=ti)

            Tpath = os.path.join(timepath,'T')
            if verbose: print('writing data to {:s}'.format(Tpath)) 
            write_data(Tpath, Tpatch.ravel(), patchName=self.name, timeName=ti)

            if k is not None:
                kpath = os.path.join(timepath,'k')
                if verbose: print('writing data to {:s}'.format(kpath)) 
                write_data(kpath, kpatch.ravel(), patchName=self.name, timeName=ti)


