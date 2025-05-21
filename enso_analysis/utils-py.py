"""
Utility functions for ENSO analysis.
"""

import pandas as pd
import numpy as np
import xarray as xr


def format_contour_label(value):
    """
    Format contour label to remove trailing zeros.
    
    Parameters
    ----------
    value : float
        Contour value
        
    Returns
    -------
    str
        Formatted contour label
    """
    return f"{value:.2f}".rstrip('0').rstrip('.')


def extract_nino_region(dataset, region, lat_range, lon_range):
    """
    Extract a Niño region from a dataset.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing SST data
    region : str
        Name of the Niño region
    lat_range : tuple
        (min_lat, max_lat) for the region
    lon_range : tuple
        (min_lon, max_lon) for the region
        
    Returns
    -------
    xarray.Dataset
        Dataset subset for the specified region
    """
    min_lat, max_lat = min(lat_range), max(lat_range)
    min_lon, max_lon = min(lon_range), max(lon_range)
    
    return dataset.sel(
        latitude=slice(max_lat, min_lat),
        longitude=slice(min_lon, max_lon)
    )


def rolling_average(dataframe, window=3, center=True):
    """
    Apply a rolling average to a dataframe.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe
    window : int, optional
        Rolling window size, defaults to 3
    center : bool, optional
        Whether to center the window, defaults to True
        
    Returns
    -------
    pandas.DataFrame
        Dataframe with rolling average applied
    """
    return dataframe.rolling(window=window, center=center).mean()


def compute_monthly_anomalies(dataset, variable, climatology_dict):
    """
    Compute monthly anomalies based on a climatology.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing the variable to compute anomalies for
    variable : str
        Name of the variable to compute anomalies for
    climatology_dict : dict
        Dictionary of climatology periods for each year
        
    Returns
    -------
    xarray.Dataset
        Dataset with anomalies
    """
    anomaly_arrays = []
    
    for time in dataset.valid_time:
        timestamp = pd.to_datetime(time.values)
        year, month = timestamp.year, timestamp.month
        period = climatology_dict.get(year)
        
        if not period:
            anomaly_arrays.append(np.nan)
            continue
            
        start_year, end_year = period
        clim_slice = dataset[variable].sel(
            valid_time=slice(f'{start_year}-01-01', f'{end_year}-12-31')
        )
        
        month_data = clim_slice.sel(
            valid_time=clim_slice['valid_time'].dt.month == month
        )
        
        if month_data.size == 0:
            anomaly_arrays.append(np.nan)
            continue
            
        clim_mean = month_data.mean(dim='valid_time')
        time_value = dataset[variable].sel(valid_time=time)
        time_value, clim_mean = xr.align(time_value, clim_mean, join='exact')
        anomaly_arrays.append(time_value - clim_mean)
        
    return xr.concat(anomaly_arrays, dim='valid_time').to_dataset(name=variable)
