# ENSO Analysis

A comprehensive toolkit for analyzing El Niño-Southern Oscillation (ENSO) patterns and their impacts on global climate using sea surface temperature (SST) data.

## Features

- **ONI Calculation**: Compute the Oceanic Niño Index following standard methodologies
- **ENSO Event Detection**: Identify and categorize El Niño and La Niña events by intensity
- **Transition Analysis**: Analyze the evolution and transitions between ENSO states
- **Visualization**: Create publication-ready visualizations of ENSO patterns

## Installation

```bash
# Clone the repository
git clone https://github.com/Rezaian/enso-analysis.git
cd enso-analysis

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
from enso_analysis import ONIAnalyzer, TransitionAnalyzer, ENSOVisualizer

# Initialize the ONI analyzer
sst_file = './data/era5_sst_1940_2024.nc'
output_dir = './output'
oni_analyzer = ONIAnalyzer(sst_file, output_dir)

# Run the full ONI analysis
oni_analyzer.load_sst_data()
oni_analyzer.generate_climatology_periods(1940, 2024)
oni_analyzer.process_nino34_data()
oni_analyzer.compute_full_oni()

# Get the events summary
events = oni_analyzer.get_event_summary()
print(f"Found {len(events)} ENSO events")

# Visualize ONI phases
visualizer = ENSOVisualizer(output_dir)
visualizer.plot_oni_phases(oni_analyzer.oni_df, oni_analyzer.elnino_thresholds)
```

## Documentation

For more detailed information on the API and usage examples, please see the [documentation](docs/README.md) and the example notebooks in the `notebooks/` directory.

## Data Requirements

The analysis requires SST data in NetCDF format. The data should contain:
- Sea Surface Temperature ('sst') variable
- Latitude and longitude coordinates
- Time dimension

Sample datasets can be downloaded from sources like ERA5 or NOAA.

## Citation

If you use this toolkit in your research, please cite it as:

```
Reza Rezaian. (2025). ENSO Analysis: A Python toolkit for ENSO analysis. 
https://github.com/Rezaian/enso-analysis
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
