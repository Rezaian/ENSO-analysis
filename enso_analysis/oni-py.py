"""
ONI (Oceanic Niño Index) analysis module.
"""

import os
import pandas as pd
import numpy as np
import xarray as xr

from .base import ENSOBaseAnalyzer
from .config import ENSOThresholds, NINO_REGIONS


class ONIAnalyzer(ENSOBaseAnalyzer):
    """
    Analyzer for calculating the Oceanic Niño Index (ONI) and identifying ENSO events.
    
    Parameters
    ----------
    sst_file : str
        Path to the SST NetCDF file
    output_dir : str, optional
        Directory to save outputs, defaults to './output'
    """
    
    def __init__(self, sst_file, output_dir='./output'):
        """Initialize the ONI analyzer."""
        super().__init__(sst_file, output_dir)
        self.elnino_thresholds = ENSOThresholds()
        self.lanina_thresholds = ENSOThresholds()
        self.oni_ds = None
        self.oni_df = None
        self.sst_nino34_mean_rol3m = None
    
    def process_nino34_data(self):
        """
        Process Niño 3.4 region data and compute 3-month rolling average.
        
        Returns
        -------
        self : ONIAnalyzer
            For method chaining
        """
        # Extract Niño 3.4 region (5°N-5°S, 170°W-120°W)
        lat_range = NINO_REGIONS['nino3.4']['latitude']
        lon_range = NINO_REGIONS['nino3.4']['longitude']
        
        sst_nino34 = self.sst_ds['sst'].sel(
            latitude=slice(lat_range[0], lat_range[1]), 
            longitude=slice(lon_range[0], lon_range[1])
        )
        
        # Compute spatial average across Niño 3.4 region
        sst_nino34_mean = sst_nino34.mean(('latitude', 'longitude')).to_dataframe()
        
        # Resample to monthly data
        sst_monthly = sst_nino34_mean['sst'].resample('M').mean()
        sst_monthly.index = sst_monthly.index.to_period('M').to_timestamp()
        sst_monthly = sst_monthly.reset_index()
        
        # Compute 3-month rolling average
        self.sst_nino34_mean_rol3m = (pd.DataFrame({
            'valid_time': sst_monthly['valid_time'],
            'sst': sst_monthly['sst'].rolling(window=3, center=True).mean()
        }).dropna().set_index('valid_time').to_xarray())
        
        return self
    
    def compute_oni(self):
        """
        Compute the Oceanic Niño Index (ONI).
        
        Returns
        -------
        xarray.Dataset
            Dataset with ONI values
        """
        sst = self.sst_nino34_mean_rol3m['sst']
        valid_times = self.sst_nino34_mean_rol3m['valid_time']
        sst_clim_avg = []
        
        for time in valid_times:
            timestamp = pd.to_datetime(time.values)
            year, month = timestamp.year, timestamp.month
            period = self.climatology_dict.get(year)
            
            if not period:
                sst_clim_avg.append(np.nan)
                continue
                
            start_year, end_year = period
            clim_slice = sst.sel(valid_time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))
            clim_month = clim_slice.sel(valid_time=clim_slice['valid_time'].dt.month == month)
            
            mean_sst = clim_month.mean().item() if clim_month.size > 0 else np.nan
            sst_clim_avg.append(mean_sst)
        
        dataset = self.sst_nino34_mean_rol3m.assign(sst_clim_avg=('valid_time', sst_clim_avg))
        self.oni_ds = dataset.assign(ONI=dataset['sst'] - dataset['sst_clim_avg'])
        return self.oni_ds
    
    def compute_full_oni(self):
        """
        Compute ONI and perform complete analysis of ENSO events.
        
        Returns
        -------
        self : ONIAnalyzer
            For method chaining
        """
        self.oni_ds = self.compute_oni()
        self.oni_df = self.oni_ds[['ONI']].to_dataframe().reset_index()
        self.oni_df = self.identify_enso_phases(self.oni_df)
        self.oni_df = self.number_sustained_events(self.oni_df)
        self.oni_df = self.categorize_events(self.oni_df)
        
        # Save ONI data
        output_path = os.path.join(self.output_dir, 'oni_data.csv')
        self.oni_df.to_csv(output_path, index=False)
        
        return self
    
    def identify_enso_phases(self, df):
        """
        Identify ENSO phases (El Niño, La Niña, or Neutral).
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with ONI values
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with ENSO phases identified
        """
        df['Phase'] = 'Neutral'
        for i in range(len(df) - 4):
            oni_values = df['ONI'].iloc[i:i+5]
            if (oni_values >= 0.5).all():
                df.loc[df.index[i:i+5], 'Phase'] = 'El Niño'
            elif (oni_values <= -0.5).all():
                df.loc[df.index[i:i+5], 'Phase'] = 'La Niña'
        return df
    
    def number_sustained_events(self, df):
        """
        Number sustained ENSO events for tracking.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with ENSO phases
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with numbered events
        """
        for col in ['ElNino_event', 'LaNina_event', 'Neutral_event']:
            df[col] = pd.Series(dtype='Int64')
        
        event_counters = {'El Niño': 0, 'La Niña': 0, 'Neutral': 0}
        current_phase = None
        
        for idx, row in df.iterrows():
            phase = row['Phase']
            if phase != current_phase:
                if phase in event_counters:
                    event_counters[phase] += 1
                current_phase = phase
            
            if phase == 'El Niño':
                df.at[idx, 'ElNino_event'] = event_counters[phase]
            elif phase == 'La Niña':
                df.at[idx, 'LaNina_event'] = event_counters[phase]
            elif phase == 'Neutral':
                df.at[idx, 'Neutral_event'] = event_counters[phase]
        
        return df
    
    def categorize_events(self, df):
        """
        Categorize ENSO events by intensity.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with ENSO events
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with categorized events
        """
        categories = {
            self.elnino_thresholds.very_strong: 'Very Strong',
            self.elnino_thresholds.strong: 'Strong',
            self.elnino_thresholds.moderate: 'Moderate',
            self.elnino_thresholds.weak: 'Weak'
        }
        
        def get_intensity(oni_values, phase):
            for threshold in sorted(categories.keys(), reverse=True):
                condition = oni_values >= threshold if phase == 'El Niño' else oni_values <= -threshold
                if pd.Series(condition.astype(int)).rolling(window=3, min_periods=3).sum().max() >= 3:
                    return categories[threshold]
            return 'Weak'
        
        for event_type, phase in [('ElNino_event', 'El Niño'), ('LaNina_event', 'La Niña')]:
            for event_num in df[event_type].dropna().unique():
                event_mask = df[event_type] == event_num
                df.loc[event_mask, 'Intensity'] = get_intensity(df.loc[event_mask, 'ONI'].values, phase)
        
        df.loc[df['Phase'] == 'Neutral', 'Intensity'] = ''
        return df
    
    def get_event_summary(self):
        """
        Summarize ENSO events, including start/end dates and intensity.
        
        Returns
        -------
        pandas.DataFrame
            Summary of ENSO events
        """
        df_filtered = self.oni_df[self.oni_df['Phase'].isin(['El Niño', 'La Niña'])].copy().sort_values('valid_time').reset_index(drop=True)
        events, current_event = [], None
        
        for idx, row in df_filtered.iterrows():
            phase, intensity, date = row['Phase'], row['Intensity'], row['valid_time']
            
            if current_event is None:
                current_event = {'start_date': date, 'end_date': date, 'phase': phase, 'intensity': intensity}
            else:
                same_event = (date == current_event['end_date'] + pd.DateOffset(months=1) and 
                             phase == current_event['phase'] and intensity == current_event['intensity'])
                if same_event:
                    current_event['end_date'] = date
                else:
                    events.append(current_event)
                    current_event = {'start_date': date, 'end_date': date, 'phase': phase, 'intensity': intensity}
        
        if current_event:
            events.append(current_event)
        
        summary_df = pd.DataFrame(events)
        summary_df['Period'] = list(zip(
            summary_df['start_date'].dt.strftime('%Y-%m'),
            summary_df['end_date'].dt.strftime('%Y-%m')
        ))
        
        # Save event summary
        output_path = os.path.join(self.output_dir, 'oni_events.csv')
        summary_df.to_csv(output_path, index=False)
        
        return summary_df[['Period', 'phase', 'intensity']].rename(
            columns={'phase': 'Phase', 'intensity': 'Intensity'}
        )
        