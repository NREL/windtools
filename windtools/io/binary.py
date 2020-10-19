# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
import struct


class BinaryFile:
    """
    Helper class for handling binary file I/O
    """
    def __init__(self,path,mode='rb'):
        """Create binary file reader object

        Sample Usage
        ------------
        ```
        from windtools.io.binary import BinaryFile
        with BinaryFile(fname) as f:
            i = f.read_int4()
            f = f.read_float()
            someArray = f.read_float(10)
        ```
        """
        self.path = path
        self.mode = mode.strip('b')
        self.f = open(path,self.mode+'b')

    def __enter__(self):
        # Called when used with the 'with' statement
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Cleandown code, called when used with the 'with' statement
            ref: http://stackoverflow.com/questions/22417323/how-do-enter-and-exit-work-in-python-decorator-classes
        """
        # clean up
        self.f.close()
        # handle exceptions
        if exc_type is not None:
            print(exc_type, exc_value, traceback)
           #return False # uncomment to pass exception through
        return self

    def close(self):
        self.f.close()

    def read(self,N=1):
        return self.f.read(N)

    def unpack(self,*args):
        try:
            return struct.unpack(*args)
        except struct.error:
            raise IOError

    def read_char(self):
        return self.f.read(1).decode('utf-8')

    def readline(self):
        s = ''
        b = self.read_char()
        while not b == '\n':
            s += b
            b = self.f.read(1).decode('utf-8')
        return s + b

    # integers
    def read_int1(self,N=1):
        if N==1: return self.unpack('b',self.f.read(1))[0] #short
        else: return self.unpack('{:d}b',self.f.read(N*1))[0:N] #short
    def read_int2(self,N=1):
        if N==1: return self.unpack('h',self.f.read(2))[0] #short
        else: return self.unpack('{:d}h'.format(N),self.f.read(N*2))[0:N] #short
    def read_int4(self,N=1):
        if N==1: return self.unpack('i',self.f.read(4))[0] #int
        else: return self.unpack('{:d}i'.format(N),self.f.read(N*4))[0:N] #int
    def read_int8(self,N=1):
        if N==1: return self.unpack('l',self.f.read(8))[0] #long
        else: return self.unpack('{:d}l'.format(N),self.f.read(N*8))[0:N] #long

    # floats
    def read_float(self,N=1,dtype=float):
        if N==1: return dtype( self.unpack('f',self.f.read(4))[0] )
        else: return [ dtype(val) for val in self.unpack('{:d}f'.format(N),self.f.read(N*4))[0:N] ]
    def read_double(self,N=1):
        if N==1: return self.unpack('d',self.f.read(8))[0]
        else: return self.unpack('{:d}d'.format(N),self.f.read(N*8))[0:N]
    def read_real4(self,N=1):
        return self.read_float(N,dtype=np.float32)
    def read_real8(self,N=1):
        return self.read_float(N,dtype=np.float64)

    # binary output
    def write_type(self,val,type):
        if hasattr(val,'__iter__'):
            N = len(val)
            self.f.write(struct.pack('{:d}{:s}'.format(N,type),*val))
        else:
            self.f.write(struct.pack(type,val))


    # aliases
    def read_int(self,N=1): return self.read_int4(N)

    def write_int1(self,val): self.write_type(val,'b')
    def write_int2(self,val): self.write_type(val,'h')
    def write_int4(self,val): self.write_type(val,'i')
    def write_int8(self,val): self.write_type(val,'l')

    def write_int(self,val): self.write_int4(val)
    def write_float(self,val): self.write_type(val,'f')
    def write_double(self,val): self.write_type(val,'d')

