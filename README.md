# ENSO Analysis

A comprehensive toolkit for analyzing El Niño-Southern Oscillation (ENSO) patterns and their impacts on global climate using sea surface temperature (SST) data.

## Features

- **ONI Calculation**: Compute the Oceanic Niño Index following standard methodologies
- **ENSO Event Detection**: Identify and categorize El Niño and La Niña events by intensity
- **Transition Analysis**: Analyze the evolution and transitions between ENSO states
- **Visualization**: Create publication-ready visualizations of ENSO patterns

## Repository Structure

```
enso-analysis/
├── LICENSE                     # MIT License file
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies
├── setup.py                    # Package installation configuration
├── run_analysis.sh             # Shell script to run the analysis
├── .gitignore                  # Git ignore file
├── data/                       # Directory for input data
│   └── .gitkeep                # Empty file to ensure directory is tracked by git
├── output/                     # Directory for analysis outputs
│   └── .gitkeep                # Empty file to ensure directory is tracked by git
├── notebooks/                  # Jupyter notebooks for examples
│   └── examples.ipynb          # Example notebook demonstrating package usage
└── enso_analysis/              # Main package directory
    ├── __init__.py             # Package initialization
    ├── __main__.py             # Command-line interface main script
    ├── base.py                 # Base classes for ENSO analysis
    ├── cli.py                  # Command-line interface
    ├── config.py               # Configuration settings
    ├── oni.py                  # ONI (Oceanic Niño Index) analysis
    ├── transition.py           # ENSO transition analysis
    ├── utils.py                # Utility functions
    └── visualization.py        # Visualization functions
```

## Key Components

1. **Core Modules:**
   - `base.py`: Base classes with common functionality
   - `oni.py`: Oceanic Niño Index calculation and event identification
   - `transition.py`: Analysis of ENSO event transitions
   - `visualization.py`: Plotting functions for ENSO data

2. **Command-Line Interface:**
   - `__main__.py`: Main CLI implementation
   - `cli.py`: Entry point for CLI commands
   - An entry point in `setup.py` to install the CLI tool

3. **Documentation:**
   - `README.md`: Comprehensive project documentation
   - Docstrings in all modules and functions
   - Example notebook demonstrating usage

4. **Utilities:**
   - `config.py`: Configuration parameters
   - `utils.py`: Helper functions

## Benefits of This Structure

1. **Modularity**: Each component has a clear responsibility, making the code easier to maintain and extend.
2. **Reusability**: Functions are organized into logical modules that can be imported and used independently.
3. **Professionalism**: Follows Python package best practices with proper documentation and installation.
4. **Extensibility**: Easy to add new functionality by creating new modules or extending existing ones.
5. **Usability**: Can be used as a library, from the command line, or through example notebooks.

## Usage Options

1. **As a Library:**
   ```python
   from enso_analysis import ONIAnalyzer, TransitionAnalyzer, ENSOVisualizer
   
   oni_analyzer = ONIAnalyzer('path/to/sst_file.nc', 'output')
   oni_analyzer.compute_full_oni()
   ```

2. **From Command Line:**
   ```bash
   # After installation
   enso-analysis --sst-file data/era5_sst_1940_2024.nc --output-dir output
   
   # Or using the script
   ./run_analysis.sh
   ```

3. **Using the Example Notebook:**
   - Open `notebooks/examples.ipynb` in Jupyter
   - Follow the step-by-step examples

## Next Steps

1. **Add Unit Tests**: Create tests for each module to ensure reliability.
2. **Add More Documentation**: Consider creating more detailed documentation with examples.
3. **Extend Functionality**: Add more analysis methods or visualization options.
4. **CI/CD Pipeline**: Set up automated testing and deployment.

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
