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

def fit_powerlaw(df=None,z=None,U=None,zref=80.0,Uref=None):
    """Calculate power-law exponent to estimate shear.

    Parameters
    ==========
    df : pd.DataFrame, optional
        Calculate from data columns; index should be height values
    U : str or array-like, optional
        An array of wind speeds if dataframe 'df' is not provided speeds
    z : array-like, optional
        An array of heights if dataframe 'df' is not provided
    zref : float
        Power-law reference height
    Uref : float, optional
        Power-law reference wind speed; if not specified, then the wind
        speeds are evaluatecd at zref to get Uref

    Returns
    =======
    alpha : float or pd.Series
        Shear exponents
    R2 : float or pd.Series
        Coefficients of determination
    """
    from scipy.optimize import curve_fit
    # generalize all inputs
    if df is None:
        assert (U is not None) and (z is not None)
        df = pd.DataFrame(U, index=z)
    elif isinstance(df,pd.Series):
        df = pd.DataFrame(df)
    # make sure we're only working with above-ground values
    df = df.loc[df.index > 0]
    z = df.index
    logz = np.log(z) - np.log(zref)
    # evaluate Uref at zref, if needed
    if Uref is None:
        Uref = df.loc[zref]
    elif not hasattr(Uref, '__iter__'):
        Uref = pd.Series(Uref,index=df.columns)
    # calculate shear coefficient
    alpha = pd.Series(index=df.columns)
    R2 = pd.Series(index=df.columns)
    def fun(x,*popt):
        return popt[0]*x
    for col,U in df.iteritems():
        logU = np.log(U) - np.log(Uref[col])
        popt, pcov = curve_fit(fun,xdata=logz,ydata=logU,p0=0.14,bounds=(0,1))
        alpha[col] = popt[0]
        U = df[col]
        resid = U - Uref[col]*(z/zref)**alpha[col]
        SSres = np.sum(resid**2)
        SStot = np.sum((U - np.mean(U))**2)
        R2[col] = 1.0 - (SSres/SStot)
    return alpha.squeeze(), R2.squeeze()


def covariance(a,b,interval='10min',resample=False,**kwargs):
    """Calculate covariance between two series (with datetime index) in
    the specified interval, where the interval is defined by a pandas
    offset string
    (http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects).

    Notes:
    - The output data will have the same length as the input data by
      default, because statistics are calculated with pd.rolling(). To
      return data at the same intervals as specified, set
      `resample=True`.
    - Covariances may be simultaneously calculated at multiple heights
      by inputting multi-indexed dataframes (with height being the
      second index level)
    - If the inputs have multiindices, this function will return a
      stacked, multi-indexed dataframe.

    Example:
        heatflux = covariance(df['Ts'],df['w'],'10min')
    """
    # handle xarray data arrays
    if isinstance(a, xr.DataArray):
        a = a.to_pandas()
    if isinstance(b, xr.DataArray):
        b = b.to_pandas()
    # handle multiindices
    have_multiindex = False
    if isinstance(a.index, pd.MultiIndex):
        assert isinstance(b.index, pd.MultiIndex), \
               'Both a and b should have multiindices'
        assert len(a.index.levels) == 2
        assert len(b.index.levels) == 2
        # assuming levels 0 and 1 are time and height, respectively
        a = a.unstack() # create unstacked copy
        b = b.unstack() # create unstacked copy
        have_multiindex = True
    elif isinstance(b.index, pd.MultiIndex):
        raise AssertionError('Both a and b should have multiindices')
    # check index
    if isinstance(interval, str):
        # make sure we have a compatible index
        assert isinstance(a.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex))
        assert isinstance(b.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex))
    # now, do the calculations
    if resample:
        a_mean = a.resample(interval).mean()
        b_mean = b.resample(interval).mean()
        ab_mean = (a*b).resample(interval,**kwargs).mean()
    else:
        a_mean = a.rolling(interval).mean()
        b_mean = b.rolling(interval).mean()
        ab_mean = (a*b).rolling(interval,**kwargs).mean()
    cov = ab_mean - a_mean*b_mean
    if have_multiindex:
        return cov.stack()
    else:
        return cov
