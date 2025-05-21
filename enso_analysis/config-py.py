"""
Configuration settings for ENSO Analysis.
"""

import os
from dataclasses import dataclass

# Default paths
DEFAULT_OUTPUT_DIR = './output'
DEFAULT_DATA_DIR = './data'

# Region definitions
NINO_REGIONS = {
    'nino1+2': {'latitude': (0, -10), 'longitude': (-90, -80)},
    'nino3': {'latitude': (5, -5), 'longitude': (-150, -90)},
    'nino3.4': {'latitude': (5, -5), 'longitude': (-170, -120)},
    'nino4': {'latitude': (5, -5), 'longitude': (160, -150)},
}

# Plotting parameters
PLOT_PARAMS = {
    'colorbar_label': 'SST Anomaly (°C)',
    'cmap': 'RdBu_r',
    'contour_levels': [i * 0.25 for i in range(-15, 15)],
    'fontsize_title': 20,
    'fontsize_ylabel': 22,
    'fontsize_tick': 20,
    'fontsize_clabel': 22,
}

@dataclass
class ENSOThresholds:
    """ENSO intensity thresholds in degrees Celsius."""
    very_strong: float = 2.0
    strong: float = 1.5
    moderate: float = 1.0
    weak: float = 0.5
