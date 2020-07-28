"""
Classes for organizing data series stored in different directory and
subdirectory structures
"""
import os

def pretty_list(strlist,indent=2,sep='\t',width=80):
    """For formatting long lists of strings of arbitrary length
    """
    sep = sep.expandtabs()
    max_item_len = max([len(s) for s in strlist])
    items_per_line = int((width - (indent+max_item_len)) / (len(sep)+max_item_len) + 1)
    Nlines = int(len(strlist) / items_per_line)
    extraline = (len(strlist) % items_per_line) > 0
    fmtstr = '{{:{:d}s}}'.format(max_item_len)
    strlist = [ fmtstr.format(s) for s in strlist ] # pad strings so that they're all the same length
    finalline = ''
    for line in range(Nlines):
        ist = line*items_per_line
        finalline += indent*' ' + sep.join(strlist[ist:ist+items_per_line]) + '\n'
    if extraline:
        finalline += indent*' ' + sep.join(strlist[Nlines*items_per_line:]) + '\n'
    return finalline


class Series(object):
    """Object for holding general series data

    Written by Eliot Quon (eliot.quon@nrel.gov)
    """
    def __init__(self,datadir,**kwargs):
        self.datadir = os.path.abspath(datadir)
        self.filelist = []
        self.times = []
        self.verbose = kwargs.get('verbose',False)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,i):
        return self.filelist[i]

    def __iter__(self):
        self.lastfile = -1  # reset iterator index
        return self

    def __next__(self):
        if self.filelist is None:
            raise StopIteration('file list is empty')
        self.lastfile += 1
        if self.lastfile >= self.Ntimes:
            raise StopIteration
        else:
            return self.filelist[self.lastfile]

    def next(self):
        # for Python 2 compatibility
        return self.__next__()
            
    def itertimes(self):
        return zip(self.times,self.filelist)

    def trimtimes(self,tstart=None,tend=None):
        if (tstart is not None) or (tend is not None):
            if tstart is None: tstart = 0.0
            if tend is None: tend = 9e9
            selected = [ (t >= tstart) & (t <= tend) for t in self.times ]
            self.filelist = [self.filelist[i] for i,b in enumerate(selected) if b ]
            self.times = [self.times[i] for i,b in enumerate(selected) if b ]
            self.Ntimes = len(self.times)

            # for SOWFATimeSeries:
            try:
                self.dirlist = [self.dirlist[i] for i,b in enumerate(selected) if b ]
            except AttributeError:
                pass


class TimeSeries(Series):
    """Object for holding time series data in a single directory

    Written by Eliot Quon (eliot.quon@nrel.gov)

    Sample usage:
        from datatools.series import TimeSeries
        #ts = TimeSeries('/path/to/data',prefix='foo',suffix='.bar')
        ts = TimeSeries('/path/to/data',prefix='foo',suffix='.bar',
                        tstart=21200,tend=21800)
        for fname in ts:
            do_something(fname)
        for t,fname in ts.itertimes():
            do_something(t,fname)
    """

    def __init__(self,
                 datadir='.',
                 prefix='', suffix='',
                 dt=1.0, t0=0.0,
                 dirs=False,
                 tstart=None, tend=None,
                 **kwargs):
        """Collect data from specified directory, for files with a
        given prefix and optional suffix. For series with integer time
        step numbers, dt can be specified (with t0 offset) to determine
        the time for each snapshot.
        """
        super(self.__class__,self).__init__(datadir,**kwargs)
        self.dt = dt
        self.t0 = t0

        if dirs:
            def check_path(f): return os.path.isdir(f)
        else:
            def check_path(f): return os.path.isfile(f)

        if self.verbose:
            print('Retrieving time series from',self.datadir)

        for f in os.listdir(self.datadir):
            if (check_path(os.path.join(self.datadir,f))) \
                    and f.startswith(prefix) \
                    and f.endswith(suffix):
                fpath = os.path.join(self.datadir,f)
                self.filelist.append(fpath)
                val = f[len(prefix):]
                if len(suffix) > 0:
                    val = val[:-len(suffix)]
                try:
                    self.times.append(t0 + dt*float(val))
                except ValueError:
                    print('Prefix and/or suffix are improperly specified')
                    print('  attempting to cast value: '+val)
                    print('  for file: '+fpath)
                    break
        self.Ntimes = len(self.filelist)
        if self.Ntimes == 0:
            print('Warning: no matching files were found')

        # sort by output time
        iorder = [kv[0] for kv in sorted(enumerate(self.times),key=lambda x:x[1])]
        self.filelist = [self.filelist[i] for i in iorder]
        self.times = [self.times[i] for i in iorder]

        # select time range
        self.trimtimes(tstart,tend)


class SOWFATimeSeries(Series):
    """Object for holding general time series data stored in multiple
    time subdirectories, e.g., as done in OpenFOAM.

    Written by Eliot Quon (eliot.quon@nrel.gov)

    Sample usage:
        from datatools.series import SOWFATimeSeries
        ts = SOWFATimeSeries('/path/to/data',filename='U')
    """

    def __init__(self,datadir='.',filename=None,**kwargs):
        """Collect data from subdirectories, assuming that subdirs
        have a name that can be cast as a float
        """
        super(self.__class__,self).__init__(datadir,**kwargs)
        self.dirlist = []
        self.timenames = []
        self.filename = filename

        if self.verbose:
            print('Retrieving SOWFA time series from',self.datadir)

        # process all subdirectories
        subdirs = [ os.path.join(self.datadir,d)
                    for d in os.listdir(self.datadir)
                    if os.path.isdir(os.path.join(self.datadir,d)) ]
        for path in subdirs:
            dname = os.path.split(path)[-1]
            try:
                tval = float(dname)
            except ValueError:
                continue
            self.times.append(tval)
            self.dirlist.append(path)
        self.Ntimes = len(self.dirlist)
    
        # sort by output time
        iorder = [kv[0] for kv in sorted(enumerate(self.times),key=lambda x:x[1])]
        self.dirlist = [self.dirlist[i] for i in iorder]
        self.times = [self.times[i] for i in iorder]

        # check that all subdirectories contain the same files
        self.timenames = os.listdir(self.dirlist[0])
        for d in self.dirlist:
            if not os.listdir(d) == self.timenames:
                print('Warning: not all subdirectories contain the same files')
                break
        if self.verbose:
            self.outputs() # print available outputs

        # set up file list
        if filename is not None:
            self.get(filename)

        # select time range
        tstart = kwargs.get('tstart',None)
        tend = kwargs.get('tend',None)
        self.trimtimes(tstart,tend)

    def get(self,filename):
        """Update file list for iteration"""
        self.filelist = []
        for path in self.dirlist:
            fpath = os.path.join(path,filename)
            if os.path.isfile(fpath):
                self.filelist.append(fpath)
            else:
                raise IOError(fpath+' not found')

    def outputs(self,prefix=''):
        """Print available outputs for the given data directory"""
        selected_output_names = [ name for name in self.timenames if name.startswith(prefix) ]
        if self.verbose:
            if prefix:
                print('Files starting with "{}" in each subdirectory:'.format(prefix))
            else:
                print('Files in each subdirectory:')
            #print('\t'.join([ '    '+name for name in selected_output_names ]))
            print(pretty_list(sorted(selected_output_names)))
        return selected_output_names

    def __repr__(self):
        return str(self.Ntimes) + ' time subdirectories located in ' + self.datadir

