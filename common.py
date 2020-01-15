# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
import pandas as pd
import xarray as xr


def calc_wind(df=None,u='u',v='v'):
    """Calculate wind speed and direction from horizontal velocity
    components, u and v.

    Parameters
    ==========
    df : pd.DataFrame or xr.Dataset
        Calculate from data columns (pandas dataframe) or data-arrays
        (xarrays dataset) named 'u' and 'v'
    u : str or array-like
        Data name if 'df' is provided; otherwise array of x-velocities
    v : str or array-like
        Data name if 'df' is provided; otherwise array of y-velocities
    """
    if df is None:
        assert (u is not None) and (v is not None)
    elif isinstance(df,pd.DataFrame):
        assert all(velcomp in df.columns for velcomp in [u,v]), \
                'velocity components u/v not found; set u and/or v'
        u = df[u]
        v = df[v]
    elif isinstance(df,xr.Dataset):
        assert all(velcomp in df.variables for velcomp in [u,v]), \
                'velocity components u/v not found; set u and/or v'
        u = df[u]
        v = df[v]
    wspd = np.sqrt(u**2 + v**2)
    wdir = 180. + np.degrees(np.arctan2(u, v))
    return wspd, wdir

def calc_uv(df=None,wspd='wspd',wdir='wdir'):
    """Calculate velocity components from wind speed and direction.

    Parameters
    ==========
    df : pd.DataFrame or xr.Dataset
        Calculate from data columns (pandas dataframe) or data-arrays
        (xarrays dataset) named 'u' and 'v'
    wspd : str or array-like
        Data name if 'df' is provided; otherwise array of wind speeds
    wdir : str or array-like
        Data name if 'df' is provided; otherwise array of wind directions
    """
    if df is None:
        assert (wspd is not None) and (wdir is not None)
    elif isinstance(df,pd.DataFrame):
        assert all(windcomp in df.columns for windcomp in [wspd,wdir]), \
                   'wind speed/direction not found; set wspd and/or wdir'
        wspd = df[wspd]
        wdir = df[wdir]
    elif isinstance(df,xr.Dataset):
        assert all(windcomp in df.variables for windcomp in [wspd,wdir]), \
                   'wind speed/direction not found; set wspd and/or wdir'
        wspd = df[wspd]
        wdir = df[wdir]
    ang = np.radians(270. - wdir)
    u = wspd * np.cos(ang)
    v = wspd * np.sin(ang)
    return u,v



