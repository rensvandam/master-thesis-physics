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
from project.model.sample import Alexa647
from project.model.detection import Spad23
from project.model.setup import ScanningSetup

# Create fluorescent emitters
emitter = Alexa647(position=(0, 0), laser_power=1e-3)

# Configure SPAD detector
sensor = Spad23(magnification=100, dead_time=100e-9)

# Run scanning microscopy simulation
setup = ScanningSetup(sensor=sensor, emitters=[emitter])
results = setup.run_experiment(scan_time=1e-3)
```

See `project/simulations/example.py` for a complete working example.

## Project Structure

```
spad-smlm/
├── project/                    # Main codebase
│   ├── model/                  # Core modules (see model/README.md)
│   │   ├── sample.py           # Fluorescent emitter models
│   │   ├── detection.py        # SPAD sensor implementations
│   │   ├── localization.py     # Localization algorithms
│   │   ├── localization_speed.py # Numba-optimized localization
│   │   ├── ISMprocessor.py     # Image Scanning Microscopy processing
│   │   ├── coherence_*.py      # Coherence analysis tools
│   │   └── setup.py            # Experimental setup configurations
│   ├── simulations/            # Example scripts and experiments
│   │   ├── example.py          # Basic usage example
│   │   ├── run_scanning_experiment.py
│   │   └── deprecated/         # Older experimental scripts
│   ├── test/                   # Unit tests
│   └── data/                   # Configuration files (psf.json)
├── archive/                    # Historical results and meeting materials
├── requirements.txt            # Python dependencies
└── MEP_report_HeikeSmedes.pdf  # Related prior thesis for reference
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
