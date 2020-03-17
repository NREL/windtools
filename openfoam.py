# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import re


class InputFile(dict):
    """Object to parse and store openfoam input file data

    Written by Eliot Quon (eliot.quon@nrel.gov)
    
    Includes support for parsing:
    - single values, with attempted cast to float/bool
    - lists
    - dictionaries
    """
    DEBUG = False

    block_defs = [
        ('{','}',dict),
        ('(',')',list),
        ('[',']',list),
    ]
    true_values = [
        'true',
        'on',
        'yes',
    ]
    false_values = [
        'false',
        'off',
        'no',
        'none',
    ]
    special_keywords = [
        'uniform',
        'nonuniform',
        'table',
    ]

    def __init__(self,fpath,nodef=False):
        """Create a dictionary of definitions from an OpenFOAM-style
        input file.

        Inputs
        ------
        fpath : str
            Path to OpenFOAM file
        nodef : bool, optional
            If the file only contains OpenFOAM data, e.g., a table of 
            vector values to be included from another OpenFOAM file,
            then create a generic 'data' parent object to contain the
            file data.
        """
        # read full file
        with open(fpath) as f:
            lines = f.readlines()
        if nodef:
            lines = ['data ('] + lines + [')']
        # trim single-line comments and remove directives
        for i,line in enumerate(lines):
            line = line.strip()
            if line.startswith('#'):
                if self.DEBUG:
                    print('Ignoring directive:',line)
                lines[i] = ''
            else:
                idx = line.find('//')
                if idx >= 0:
                    lines[i] = line[:idx].strip()
        # trim multi-line comments
        txt = '\n'.join(lines)
        idx0 = txt.find('/*')
        while idx0 >= 0:
            idx1 = txt.find('*/',idx0+1)
            assert (idx1 > idx0), 'Mismatched comment block'
            if self.DEBUG:
                print('Remove comment block:',txt[idx0:idx1])
            txt = txt[:idx0] + txt[idx1+2:]
            idx0 = txt.find('/*')
        # consolidate definitions into single lines
        txt = txt.replace('\n',' ')
        txt = txt.replace('\t',' ')
        txt = txt.strip()
        # now parse each line
        for name,line,containertype in self._split_defs(txt):
            if self.DEBUG:
                print('\nPARSING',name,'FROM',line,'of TYPE',containertype)
            self._parse(name,line,containertype)
        self._sanitycheck()

    def _sanitycheck(self):
        """Make sure the InputFile was read properly"""
        noparent = [key is None for key in self.keys()]
        if any(noparent):
            print('Definitions improperly read, some values without keys')
            print('If you believe this is an error, then re-run with the nodef keyword')

    def _format_item_str(self,val,maxstrlen=60):
        printval = str(val)
        if isinstance(val,list) and (len(printval) > maxstrlen):
            printval = '[list of length {:d}]'.format(len(val))
        return printval

    def __repr__(self):
        descstrs = [
            '{:s} : {:s}'.format(key, self._format_item_str(val))
            for key,val in self.items()
        ]
        return '\n'.join(descstrs)

    def _split_defs(self,txt):
        """Splits blocks of text into lines in the following forms:
            key value;
            key (values...)
            key {values...}
            (values...)
            ((values...) (values...))
        where lists and dicts may be nested. The outlier case is the
        (nested) list which takes on the key of its parent.
        """
        names, lines, container = [], [], []
        while len(txt) > 0:
            if self.DEBUG:
                print('current text:',txt)

            if (txt[0] == '('):
                # special treatment for lists, or lists within a list
                name = None
            else:
                # - find first word (name)
                idx = txt.find(' ')
                name = txt[:idx]
                if self.DEBUG: print('name=',name)
                txt = txt[idx+1:].strip()

            # - find next word (either a value/block)
            idx = txt.find(' ')
            if idx < 0:
                # EOF
                string = txt
                txt = '' # to exit loop
                if self.DEBUG: print('EOF',string)
            else:
                string = txt[:idx].strip()
                if string in self.special_keywords:
                    # append special keyword to name and read the next word
                    name += '_'+string
                    txt = txt[idx+1:].strip()
                    idx = txt.find(' ')
                    assert (idx > 0), 'problem parsing '+string+' field'
                    string = txt[:idx].strip()

            if string.endswith(';'):
                # found single definition
                if self.DEBUG: print('value=',string[:-1])
                names.append(name)
                lines.append(string[:-1]) # strip ;
                container.append(None)
            else:
                # found block
                if self.DEBUG: print('current string:',string)
                blockstart = string[0]
                blockend = None
                blocktype = None
                for block in self.block_defs:
                    if blockstart == block[0]:
                        blockend = block[1]
                        blocktype = block[2]
                        break
                assert (blockend is not None), 'Unknown input block '+blockstart
                # find end of block
                idx = txt.find(blockend) + 1
                assert (idx > 0), 'Mismatched input block'
                # consolidate spaces
                blockdef = re.sub(' +',' ',txt[:idx].strip())
                Nopen = blockdef.count(blockstart)
                Nclose = blockdef.count(blockend)
                while Nopen != Nclose:
                    if self.DEBUG:
                        print('  incomplete:',blockdef)
                    idx = txt.find(blockend, idx) + 1
                    blockdef = txt[:idx].strip()
                    Nopen = blockdef.count(blockstart)
                    Nclose = blockdef.count(blockend)
                # select block
                if self.DEBUG: print('complete block=',blockdef)
                names.append(name)
                lines.append(blockdef)
                container.append(blocktype)
            if self.DEBUG: print('container type=',container[-1])
            # trim text block
            txt = txt[idx+1:].strip()

        return zip(names, lines, container)

    def _parse(self,name,defn,containertype,parent=None):
        """Parse values split up by _split_defs()

        Casts to float and bool (the latter by checking against a list
        of known true/false values, since bool(some_str) will return 
        True if the string has a nonzero length) will be attempted.

        If the value is a container (i.e., list or dict), then 
        _split_defs() and _parse() will be called recursively.
        """
        if self.DEBUG:
            print('----------- parsing block -----------')
            if parent is not None:
                print('name:',name,'parent:',str(parent))
            if containertype is not None:
                print('container type:',containertype)
        defn = defn.strip()
        if containertype is None:
            # set single value in parent 
            defn = self._try_cast(defn)
            # SET VALUE HERE
            if self.DEBUG:
                print(name,'-->',defn)
            if parent is None:
                self.__setitem__(name, defn)
            elif isinstance(parent, dict):
                parent[name] = defn
            else:
                assert isinstance(parent, list)
                parent.append(defn)
        else:
            # we have a subblock, create new container
            if parent is None:
                # parent is the InputFile object
                if self.DEBUG:
                    print('CREATING',containertype,'named',name)
                self.__setitem__(name, containertype())
                newparent = self.__getitem__(name)
            elif isinstance(parent, dict):
                # parent is a dictionary
                if self.DEBUG:
                    print('ADDING dictionary entry,',name)
                parent[name] = containertype()
                newparent = parent[name]
            else:
                assert isinstance(parent, list)
                # parent is a list
                if self.DEBUG:
                    print('ADDING list item, name=',name)
                if name is not None:
                    # if we have nested nists with mixed types we could
                    # end up here...
                    parent.append(self._try_cast(name))
                newparent = containertype()
                parent.append(newparent)
            newdefn = defn[1:-1].strip()
            if (containertype is list) \
                    and ('(' not in newdefn) and (')' not in newdefn):
                # special treatment for lists
                for val in newdefn.split():
                    # recursively call parse wihout a name (None for
                    # list) and without a container type to indicate
                    # that a new value should be set
                    self._parse(None,val,None,parent=newparent)
            else:
                for newname,newdef,newcontainertype in self._split_defs(newdefn):
                    self._parse(newname,newdef,newcontainertype,parent=newparent)

    def _try_cast(self,s):
        assert(s.find(' ') < 0)
        try:
            # attempt float cast
            s = float(s)
        except ValueError:
            # THIS IS A TRAP
            #try:
            #    # attempt boolean cast
            #    s = bool(s)
            #except ValueError:
            #    # default to string
            #    pass
            if s.lower() in self.true_values:
                s = True
            elif s.lower() in self.false_values:
                s = False
            else:
                # default to string
                s = s.strip('"')
                s = s.strip('\'')
        return s
