# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import sys
import pandas as pd
from windtools.SOWFA6.log import LogFile

outfile = 'log.csv'

if len(sys.argv) <= 1:
    sys.exit('Specify log file(s)')

print('Scraping log files:',sys.argv[1:])
df = pd.concat([LogFile(fpath).df for fpath in sys.argv[1:]])

# drop duplicate rows (for restarts)
df.drop_duplicates(keep='last',inplace=True)

print('Writing',outfile)
df.to_csv(outfile)

