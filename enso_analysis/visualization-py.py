"""
Visualization module for ENSO analysis.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.ndimage as ndi
import numpy as np

from .config import PLOT_PARAMS


class ENSOVisualizer:
    """
    Visualizer for ENSO analysis results.
    
    Parameters
    ----------
    output_dir : str, optional
        Directory to save outputs, defaults to './output'
    """
    
    def __init__(self, output_dir='./output'):
        """Initialize the ENSO visualizer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.plot_params = PLOT_PARAMS.copy()
    
    def plot_oni_phases(self, df, thresholds, end_date=None, filename='oni_phases'):
        """
        Plot ONI phases showing El Niño and La Niña events.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with ONI values and phases
        thresholds : ENSOThresholds
            ENSO intensity thresholds
        end_date : str, optional
            End date for the plot, defaults to None (use all data)
        filename : str, optional
            Base filename for the saved plot, defaults to 'oni_phases'
            
        Returns
        -------
        tuple
            (fig, ax) matplotlib objects
        """
        plot_df = df.copy()
        if end_date:
            plot_df = plot_df[plot_df['valid_time'] <= pd.Timestamp(end_date)]
        
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Plot ONI time series
        ax.plot(plot_df['valid_time'], plot_df['ONI'], color='black')
        
        # Fill El Niño periods
        ax.fill_between(
            plot_df['valid_time'], 0.5, plot_df['ONI'],
            where=(plot_df['Phase'] == 'El Niño'), 
            color='red', alpha=0.5, interpolate=True
        )
        
        # Fill La Niña periods
        ax.fill_between(
            plot_df['valid_time'], plot_df['ONI'], -0.5,
            where=(plot_df['Phase'] == 'La Niña'), 
            color='blue', alpha=0.5, interpolate=True
        )
        
        # Add threshold lines and labels
        thresholds_dict = {
            'Very Strong': thresholds.very_strong,
            'Strong': thresholds.strong,
            'Moderate': thresholds.moderate,
            'Weak': thresholds.weak
        }
        
        for label, value in thresholds_dict.items():
            # Add positive (El Niño) threshold
            ax.axhline(value, color='gray', linewidth=1, linestyle='--')
            ax.text(
                1.01, value + 0.25, label, 
                ha='left', va='center',
                transform=ax.get_yaxis_transform(), 
                color='red', fontsize=14
            )
            
            # Add negative (La Niña) threshold
            ax.axhline(-value, color='gray', linewidth=1, linestyle='--')
            ax.text(
                1.01, -value - 0.25, label, 
                ha='left', va='center',
                transform=ax.get_yaxis_transform(), 
                color='blue', fontsize=14
            )
        
        # Add baseline
        ax.axhline(0, color='black', linewidth=0.5)
        
        # Set labels and tick parameters
        ax.set_xlabel('Year', fontsize=16)
        ax.set_ylabel('ONI (°C)', fontsize=16)
        
        ax.tick_params(axis='x', which='major', labelsize=14, length=10, width=2, direction='inout', pad=10)
        ax.tick_params(axis='y', which='major', labelsize=14, length=10, width=2, direction='inout', pad=10)
        ax.tick_params(axis='x', which='minor', labelsize=14, length=5, width=1, direction='inout', pad=10)
        
        # Set axis limits
        ax.set_xlim(plot_df['valid_time'].min(), plot_df['valid_time'].max())
        ax.set_ylim(-3, 3)
        
        # Set spine properties
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        ax.margins(x=0)
        
        # Configure date ticks
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Add extra ticks for start and end years
        start_year = plot_df['valid_time'].min().year
        end_year = plot_df['valid_time'].max().year
        extra_dates = [pd.Timestamp(year=year, month=1, day=1) for year in [start_year, end_year]]
        extra_ticks_mpl = [mdates.date2num(date) for date in extra_dates]
        
        existing_ticks = ax.get_xticks()
        filtered_ticks = [
            tick for tick in sorted(set(existing_ticks.tolist() + extra_ticks_mpl)) 
            if mdates.num2date(tick).year != 2025
        ]
        
        ax.set_xticks(filtered_ticks)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, filename)
        plt.savefig(f"{plot_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{plot_path}.eps", format='eps', dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def create_longitude_time_plot(self, datasets, plot_titles, save_path, vmin=-2, vmax=2):
        """
        Create a longitude-time plot of SST anomalies.
        
        Parameters
        ----------
        datasets : list
            List of datasets to plot
        plot_titles : list
            List of plot titles
        save_path : str
            Base path for saving the plot
        vmin : float, optional
            Minimum value for the colorbar, defaults to -2
        vmax : float, optional
            Maximum value for the colorbar, defaults to 2
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
            
        Raises
        ------
        ValueError
            If the number of datasets and titles don't match
        """
        if len(datasets) != 8 or len(plot_titles) != 8:
            raise ValueError("Must provide exactly 8 datasets and 8 titles")
            
        fig, axes = plt.subplots(2, 4, figsize=(20, 12), constrained_layout=True)
        
        for idx, (ds, title) in enumerate(zip(datasets, plot_titles)):
            row, col = idx // 4, idx % 4
            ax = axes[row, col]
            
            # Extract data
            data = ds['sst'].values
            rolling_time = ds['rolling_time'].values
            custom_longitude = ds['custom_longitude'].values
            
            # Create meshgrid for plotting
            lon_grid, time_grid = np.meshgrid(custom_longitude, rolling_time)
            
            # Apply Gaussian smoothing
            data_smooth = ndi.gaussian_filter(data, sigma=(3, 6))
            
            # Create pseudocolor plot
            img = ax.pcolormesh(
                lon_grid, time_grid, data_smooth, 
                cmap=self.plot_params['cmap'],
                vmin=vmin, vmax=vmax, shading='auto'
            )
            
            # Add contours
            contours = ax.contour(
                lon_grid, time_grid, data_smooth, 
                levels=self.plot_params['contour_levels'],
                colors='white', linewidths=1.5
            )
            
            # Add contour labels
            labels = ax.clabel(
                contours, inline=True, 
                fontsize=self.plot_params['fontsize_clabel'],
                fmt=self.plot_params.get('contour_fmt', lambda x: f"{x:.2f}")
            )
            
            for label in labels:
                label.set_rotation(0)
                label.set_color('black')
            
            # Set title
            ax.set_title(
                title, 
                fontsize=self.plot_params['fontsize_title'], 
                pad=15, color='black'
            )
            
            # Add Y-axis label for leftmost plots
            if col == 0:
                ax.set_ylabel(
                    'Year [0]               Year [+1]', 
                    fontsize=self.plot_params['fontsize_ylabel'],
                    labelpad=10, color='black'
                )
            
            # Configure tick parameters
            ax.tick_params(
                axis='both', which='major', 
                labelsize=self.plot_params['fontsize_tick'],
                labelcolor='black'
            )
            
            # Set custom X-tick positions and labels
            x_ticks = [0, (60/160)*640, (120/160)*640, 640]
            x_labels = ['120°E', '180°', '120°W', '80°W']
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, color='black', fontsize=self.plot_params['fontsize_tick'])
            
            # Set custom Y-tick positions and labels
            y_ticks = [-0.5, 2.5, 5.5, 8.5, 11.5, 14.5, 17.5, 20.5, 23.5]
            y_labels = ['JAN', 'APR', 'JUL', 'OCT', 'JAN', 'APR', 'JUL', 'OCT', 'JAN']
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, color='black', fontsize=self.plot_params['fontsize_tick'])
            
            # Set spine properties
            for spine in ax.spines.values():
                spine.set_linewidth(3)
                spine.set_color('black')
            
            # Add horizontal line at January of year+1
            ax.axhline(y=11.5, linestyle='--', color='black')
            
            # Add markers at 180° and 120°W
            for x_pos in [(60/160)*640, (120/160)*640]:
                ax.plot(x_pos, -0.25, marker='v', markersize=3, 
                       markeredgecolor='black', markerfacecolor='black')
        
        # Add colorbar
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=self.plot_params['cmap'], norm=norm)
        sm.set_array([])
        
        cbar = fig.colorbar(
            sm, ax=axes.ravel().tolist(), orientation='horizontal', 
            fraction=0.046, pad=0.04, aspect=30, shrink=0.6
        )
        
        cbar.set_label(
            self.plot_params['colorbar_label'], 
            fontsize=self.plot_params['fontsize_ylabel'], 
            labelpad=5, color='black'
        )
        
        cbar.ax.tick_params(labelsize=self.plot_params['fontsize_tick'], labelcolor='black')
        
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(3)
            spine.set_color('black')
        
        # Save the plot
        plt.savefig(f"{save_path}.png", dpi=300)
        plt.savefig(f"{save_path}.eps", dpi=300, format='eps')
        
        return fig
    
    def plot_enso_transition_composites(self, transition_analyzer, save_path='lon_t_sst_plot'):
        """
        Plot ENSO transition composites for all categories.
        
        Parameters
        ----------
        transition_analyzer : TransitionAnalyzer
            Transition analyzer with calculated composites
        save_path : str, optional
            Base path for saving the plot, defaults to 'lon_t_sst_plot'
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        # Get datasets and organize them
        composites = transition_analyzer.get_composite_dataset()
        
        datasets = [
            composites.get('very_strong_el_nino', None),
            composites.get('strong_el_nino', None),
            composites.get('moderate_el_nino', None),
            composites.get('weak_el_nino', None),
            composites.get('very_strong_la_nina', None),
            composites.get('strong_la_nina', None),
            composites.get('moderate_la_nina', None),
            composites.get('weak_la_nina', None)
        ]
        
        # Create titles
        plot_titles = [
            '(a) Very Strong El Niño',
            '(b) Strong El Niño',
            '(c) Moderate El Niño',
            '(d) Weak El Niño',
            '(e) Very Strong La Niña',
            '(f) Strong La Niña',
            '(g) Moderate La Niña',
            '(h) Weak La Niña'
        ]
        
        # Create the plot
        full_save_path = os.path.join(self.output_dir, save_path)
        return self.create_longitude_time_plot(
            datasets=datasets,
            plot_titles=plot_titles,
            save_path=full_save_path
        )
