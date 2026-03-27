# SPAD-SMLM: Single Photon Avalanche Diode Super-Resolution Microscopy

A Python framework for simulating and analyzing Single Molecule Localization Microscopy (SMLM) using Single Photon Avalanche Diode (SPAD) arrays. Developed as part of a Master's thesis at Delft University of Technology. Parts of the code were sourced from Heike Smedes' code she setup during her thesis, but the main pipeline was built custom.

## Overview

This project implements a complete simulation pipeline for SPAD-based SMLM systems:

- **Forward Model**: Fluorescent emitter photon emission, SPAD detection with dead time effects
- **Localization Algorithms**: Centroid, MLE, and ISM (Image Scanning Microscopy) methods
- **Coherence Analysis**: Sub-Poissonian photon statistics and second-order coherence functions
- **Performance Evaluation**: Bias, precision, and Jaccard index metrics

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd spad-smlm

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from project.simulations.examples.run_scanning_experiment import run_scanning_experiment

# Run a scanning microscopy simulation
setup, photon_map, G2_map, n_emitters_map, metadata = run_scanning_experiment(
    emitter_density=3,      # emitters per μm²
    laser_power=50e3,       # W/cm²
    dwell_time=1.0,         # ms per position
    area_size=(2.0, 2.0),   # μm
    positions=(10, 10),     # scan grid
    show_plots=True,
    save_data=False
)

print(f"Detected {photon_map.sum():.0f} photons")
print(f"Ground truth: {len(metadata['emitter_positions'])} emitters")
```

See `project/simulations/examples/` for more examples.

## Project Structure

```
spad-smlm/
├── project/
│   ├── model/                  # Core modules
│   │   ├── sample.py           # Fluorescent emitter models
│   │   ├── detection.py        # SPAD sensor implementations
│   │   ├── localization.py     # Localization algorithms
│   │   ├── ISMprocessor.py     # Image Scanning Microscopy processing
│   │   ├── coherence_from_data.py  # Coherence analysis
│   │   ├── setup.py            # Experimental setup
│   │   └── helper_functions.py # Utilities
│   ├── simulations/
│   │   ├── examples/           # Start here
│   │   ├── evaluation/         # Localization performance analysis
│   │   ├── experiments/        # Parameter studies
│   │   └── notebooks/          # Jupyter notebooks
│   ├── test/                   # Unit tests
│   └── data/                   # Configuration (psf.json)
└── requirements.txt
```

## Key Features

### Simulation
- Realistic fluorophore models (Alexa647) with quantum properties
- SPAD arrays (23x23, 512x512) with configurable dead time
- Widefield and scanning microscopy modes
- Noise modeling and photon statistics

### Localization Methods
- Centroid-based localization
- Maximum likelihood estimation (MLE)
- ISM pixel reassignment for resolution enhancement
- Multi-emitter fitting with coherence information

### Analysis
- Second-order coherence g2(tau) calculations
- Emitter number estimation from photon statistics
- Bias and precision evaluation
- Cramér-Rao bound analysis

## Documentation

- **Model documentation**: See `project/model/README.md` for detailed module descriptions
- **Examples**: Check `project/simulations/` for usage examples
- **Notebooks**: Interactive examples in `project/simulations/*.ipynb`

## Academic Context

Developed as a Masters Project (MEP) at Delft University of Technology, exploring the intersection of quantum optics, statistical signal processing, and super-resolution microscopy.
