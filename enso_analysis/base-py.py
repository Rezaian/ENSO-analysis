"""
Base classes for ENSO analysis.
"""

import os
import warnings
import xarray as xr
import numpy as np
from dataclasses import dataclass

from .config import ENSOThresholds

# Suppress UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)

class ENSOBaseAnalyzer:
    """
    Base class for ENSO analysis with common functionality.
    
    Parameters
    ----------
    sst_file : str
        Path to the SST NetCDF file
    output_dir : str, optional
        Directory to save outputs, defaults to './output'
    """
    
    def __init__(self, sst_file, output_dir='./output'):
        """Initialize the ENSO base analyzer."""
        self.sst_file = sst_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.sst_ds = None
        self.climatology_dict = None
    
    def load_sst_data(self):
        """
        Load SST data from NetCDF file.
        
        Returns
        -------
        self : ENSOBaseAnalyzer
            For method chaining
        """
        self.sst_ds = xr.open_dataset(self.sst_file).astype('float64')
        return self
    
    def generate_climatology_periods(self, start_year, end_year):
        """
        Generate climatology periods for each year following NOAA's methodology.
        
        Parameters
        ----------
        start_year : int
            First year in the dataset
        end_year : int
            Last year in the dataset
            
        Returns
        -------
        dict
            Dictionary of climatology periods for each year
            
        Raises
        ------
        ValueError
            If the range is less than 31 years
        """
        if end_year - start_year + 1 < 31:
            raise ValueError("Range must be at least 31 years for climatology windows.")
        
        climatology_dict = {}
        fixed_start = self._calculate_climatology_window(start_year + 15)
        fixed_end = self._calculate_climatology_window(end_year - 15)
        
        for year in range(start_year, end_year + 1):
            if start_year + 15 <= year <= end_year - 15:
                climatology_dict[year] = self._calculate_climatology_window(year)
            elif year < start_year + 15:
                climatology_dict[year] = fixed_start
            else:
                climatology_dict[year] = fixed_end
        
        self.climatology_dict = climatology_dict
        return climatology_dict
    
    @staticmethod
    def _calculate_climatology_window(year):
        """
        Calculate the 30-year window for a given year.
        
        Parameters
        ----------
        year : int
            Center year for climatology window
            
        Returns
        -------
        tuple
            (start_year, end_year) for the climatology window
        """
        rounded_year = (year // 5) * 5
        return rounded_year - 14, rounded_year + 15
