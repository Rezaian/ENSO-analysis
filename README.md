# ENSO-analysis

# ENSO Analysis

A comprehensive toolkit for analyzing El Niño-Southern Oscillation (ENSO) patterns using SST data.

## Repository Structure

```
enso-analysis/
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── data/
│   └── .gitkeep
├── output/
│   └── .gitkeep
├── notebooks/
│   └── examples.ipynb
└── enso_analysis/
    ├── __init__.py
    ├── config.py
    ├── base.py
    ├── oni.py
    ├── transition.py
    ├── visualization.py
    └── utils.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/rezaian/enso-analysis.git
cd enso-analysis

# Install the package
pip install -e .
```

## Usage

```python
from enso_analysis import ONIAnalyzer, TransitionAnalyzer, ENSOVisualizer

# Initialize analyzers
oni_analyzer = ONIAnalyzer('path/to/sst_file.nc', 'output')
oni_analyzer.run_analysis()

# See example notebooks for more usage scenarios
```

## Features

- Oceanic Niño Index (ONI) calculation
- ENSO event identification and categorization
- Transition pattern analysis
- Visualizations of ENSO phases and transitions
