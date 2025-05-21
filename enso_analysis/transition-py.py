"""
Transition analysis module for studying ENSO event transitions.
"""

import os
import pandas as pd
import numpy as np
import xarray as xr

from .base import ENSOBaseAnalyzer
from .utils import compute_monthly_anomalies


class TransitionAnalyzer(ENSOBaseAnalyzer):
    """
    Analyzer for ENSO transition patterns.
    
    Parameters
    ----------
    sst_file : str
        Path to the SST NetCDF file
    oni_file : str
        Path to the ONI data CSV file
    output_dir : str, optional
        Directory to save outputs, defaults to './output'
    """
    
    def __init__(self, sst_file, oni_file, output_dir='./output'):
        """Initialize the Transition analyzer."""
        super().__init__(sst_file, output_dir)
        self.oni_file = oni_file
        self.oni_df = None
        self.sst_monthly = None
        self.sst_anomalies = None
        self.transition_years = None
        self.transition_sst_ds = {}
        self.sst_composites = {}
        self.pacific_sst_composites = {}
    
    def load_oni_data(self):
        """
        Load ONI data from CSV file.
        
        Returns
        -------
        self : TransitionAnalyzer
            For method chaining
        """
        self.oni_df = pd.read_csv(self.oni_file)
        self.oni_df['valid_time'] = pd.to_datetime(self.oni_df['valid_time'])
        return self
    
    def process_monthly_sst(self):
        """
        Process monthly SST data.
        
        Returns
        -------
        self : TransitionAnalyzer
            For method chaining
        """
        # First, average over latitude to get a zonal mean
        sst_mean = self.sst_ds.mean(dim='latitude')
        
        # Convert to monthly data
        monthly_times = pd.to_datetime(sst_mean.valid_time.values).to_period('M').to_timestamp()
        sst_mean = sst_mean.assign_coords(valid_time=monthly_times)
        self.sst_monthly = sst_mean.groupby('valid_time').mean(dim='valid_time')
        self.sst_monthly = self.sst_monthly.assign_coords(valid_time=pd.to_datetime(self.sst_monthly.valid_time))
        
        return self
    
    def calculate_monthly_anomalies(self):
        """
        Calculate monthly SST anomalies.
        
        Returns
        -------
        self : TransitionAnalyzer
            For method chaining
        """
        self.sst_anomalies = compute_monthly_anomalies(self.sst_monthly, 'sst', self.climatology_dict)
        return self
    
    def identify_enso_transitions(self):
        """
        Identify years with ENSO transitions.
        
        Returns
        -------
        dict
            Dictionary of transition years by category
        """
        phases = ['El Niño', 'La Niña']
        intensities = ['Very Strong', 'Strong', 'Moderate', 'Weak']
        transition_years = {}
        
        for phase in phases:
            for intensity in intensities:
                category = f"{intensity.lower().replace(' ', '_')}_{phase.lower().replace(' ', '_').replace('ñ', 'n')}"
                transition_years[category] = self._get_transition_years(phase, intensity)
                
        self.transition_years = transition_years
        
        # Save transition years to a file
        output_path = os.path.join(self.output_dir, 'transition_years.csv')
        
        # Convert to DataFrame for easier saving
        transition_data = []
        for category, years in transition_years.items():
            for year in years:
                transition_data.append({'category': category, 'year': year})
        
        pd.DataFrame(transition_data).to_csv(output_path, index=False)
        
        return transition_years
    
    def _get_transition_years(self, phase, intensity):
        """
        Get years when transitions to a specific ENSO phase and intensity occurred.
        
        Parameters
        ----------
        phase : str
            ENSO phase ('El Niño' or 'La Niña')
        intensity : str
            ENSO intensity ('Very Strong', 'Strong', 'Moderate', or 'Weak')
            
        Returns
        -------
        list
            List of transition years
        """
        df = self.oni_df.copy()
        df['prev_Phase'] = df['Phase'].shift(1)
        
        condition = ((df['Phase'] == phase) & 
                     (df['prev_Phase'].isin(['Neutral', 'La Niña' if phase == 'El Niño' else 'El Niño'])) &
                     (df['Intensity'] == intensity))
        
        return df.loc[condition, 'valid_time'].dt.year.tolist()
    
    def create_transition_datasets(self):
        """
        Create datasets for transition analysis.
        
        Returns
        -------
        self : TransitionAnalyzer
            For method chaining
        """
        for category, years in self.transition_years.items():
            if years:
                self.transition_sst_ds[category] = self._create_24month_dataset(self.sst_anomalies, years, 'sst')
        
        return self
    
    def _create_24month_dataset(self, dataset, years, variable):
        """
        Create a 24-month dataset for transition analysis.
        
        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset with the variable to analyze
        years : list
            List of transition years
        variable : str
            Variable name to extract
            
        Returns
        -------
        xarray.Dataset
            Dataset with 24 months of data for each transition year
        """
        longitude = dataset.longitude.values
        n_years = len(years)
        data_array = np.full((n_years, 24, len(longitude)), np.nan, dtype=np.float32)
        dates_array = np.full((n_years, 24), np.datetime64('NaT'), dtype='datetime64[ns]')
        valid_years = []
        
        for idx, year in enumerate(years):
            start_date = pd.Timestamp(year=year, month=1, day=1)
            end_date = pd.Timestamp(year=year+1, month=12, day=31)
            data_subset = dataset.sel(valid_time=slice(start_date, end_date))
            
            if len(data_subset.valid_time) == 0:
                continue
                
            n_months = min(24, len(data_subset.valid_time))
            data_array[idx, :n_months, :] = data_subset[variable].values[:n_months]
            dates = pd.date_range(start=start_date, periods=n_months, freq='MS')
            dates_array[idx, :n_months] = dates.values
            valid_years.append(year)
        
        if len(valid_years) < n_years:
            data_array = data_array[:len(valid_years), :, :]
            dates_array = dates_array[:len(valid_years), :]
        
        return xr.Dataset(
            {variable: (['transition_year', 'month', 'longitude'], data_array)},
            coords={
                'transition_year': valid_years,
                'month': np.arange(24),
                'longitude': longitude,
                'date': (['transition_year', 'month'], dates_array)
            }
        )
    
    def calculate_composite_means(self):
        """
        Calculate composite means for each transition category.
        
        Returns
        -------
        self : TransitionAnalyzer
            For method chaining
        """
        for category, ds in self.transition_sst_ds.items():
            self.sst_composites[category] = ds.sst.mean(dim='transition_year', skipna=True)
        
        return self
    
    def spatial_slice_pacific(self, dataarray, variable):
        """
        Create a spatial slice of the Pacific Ocean.
        
        Parameters
        ----------
        dataarray : xarray.DataArray or xarray.Dataset
            Data to slice
        variable : str
            Variable name if dataarray is a Dataset
            
        Returns
        -------
        xarray.Dataset
            Dataset with Pacific Ocean slice
        """
        if isinstance(dataarray, xr.Dataset):
            dataarray = dataarray[variable]
            
        # Get western and eastern Pacific
        west_pacific = dataarray.sel(longitude=slice(120, 180))
        east_pacific = dataarray.sel(longitude=slice(-180, -80))
        
        # Concatenate to get complete Pacific basin
        pacific = xr.concat([west_pacific, east_pacific], dim='longitude')
        idx = np.arange(pacific.sizes['longitude'])
        
        # Assign consistent coordinates
        pacific = pacific.assign_coords(
            custom_longitude=('longitude', idx),
            number=('longitude', idx)
        )
        
        # Rename dimensions for consistency
        if 'month' in pacific.dims:
            pacific = pacific.rename({'month': 'rolling_time'})
        
        # Ensure correct data types
        pacific['rolling_time'] = pacific['rolling_time'].astype('int64')
        pacific['number'] = pacific['number'].astype('int64')
        pacific['custom_longitude'] = pacific['custom_longitude'].astype('int64')
        
        return pacific.transpose('rolling_time', 'longitude').to_dataset(name=variable)
    
    def prepare_pacific_composites(self):
        """
        Prepare Pacific composites for visualization.
        
        Returns
        -------
        self : TransitionAnalyzer
            For method chaining
        """
        for category, comp in self.sst_composites.items():
            self.pacific_sst_composites[category] = self.spatial_slice_pacific(comp, 'sst')
        
        return self
    
    def run_transition_analysis(self):
        """
        Run the complete transition analysis pipeline.
        
        Returns
        -------
        self : TransitionAnalyzer
            For method chaining
        """
        self.load_oni_data()
        self.process_monthly_sst()
        self.calculate_monthly_anomalies()
        self.identify_enso_transitions()
        self.create_transition_datasets()
        self.calculate_composite_means()
        self.prepare_pacific_composites()
        
        return self
    
    def get_composite_dataset(self):
        """
        Get the complete composite dataset for visualization.
        
        Returns
        -------
        dict
            Dictionary of Pacific SST composites by category
        """
        return self.pacific_sst_composites
