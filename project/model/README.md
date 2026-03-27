# Model Module

Core modeling components for the SPAD-SMLM simulation framework.

## Core Files

### `sample.py`
**Fluorescent Emitter Models**
- Fluorescent emitter classes (e.g., `Alexa647`) with realistic quantum properties
- Photon emission statistics, excitation/emission rates, and quantum yield
- Excitation time simulation using exponential distributions

### `detection.py`
**SPAD Sensor Implementation**
- Abstract `Sensor` base class and implementations (`Spad23`, `Spad512`)
- SPAD array geometries with configurable pixel arrangements
- Dead time effects and photon detection efficiency
- Visualization functions for photon detection (`show_photons`)

### `localization.py`
**Localization Algorithms**
- Centroid-based localization
- Maximum likelihood estimation (MLE)
- Weighted centroid methods
- Bias and precision analysis
- Multi-emitter fitting with Gaussian models

### `setup.py`
**Experimental Setup Modeling**
- `ScanningSetup` class for scanning microscopy configurations
- Widefield and scanning illumination modes
- Magnification and system parameter management
- Integration with ISM processing and coherence analysis

## Analysis and Processing

### `coherence_from_data.py`
**Photon Statistics and Coherence Analysis**
- Second-order quantum coherence functions g²(τ)
- Auto-coherence and cross-coherence measurements
- Sub-Poissonian photon statistics analysis
- Emitter number estimation from coherence measurements

### `ISMprocessor.py`
**Image Scanning Microscopy (ISM) Processing**
- Pixel reassignment algorithms for resolution enhancement
- Shift vector calculation and image registration
- Configurable reassignment parameter (α)

### `helper_functions.py`
**Utility Functions**
- Array merging algorithms for photon data (`merge_k`, `merge_k_2D`)
- Gaussian fitting utilities for PSF modeling
- Center of mass calculations
- Hash generation for reproducible simulations

## Module Dependencies

```
sample.py           → Emitter photon generation
      ↓
detection.py        → SPAD sensor simulation
      ↓
setup.py            → Experiment orchestration
      ↓
localization.py     → Position estimation
coherence_from_data.py → Photon statistics
ISMprocessor.py     → Resolution enhancement
```

## Deprecated

See `deprecated/` folder for unused modules preserved for reference:
- `localization_speed.py` - Numba-optimized localization (never integrated)
- `coherence_analytical.py` - Analytical coherence expressions
- `emitter_density_map.py` - Spatial density estimation
- `plot_functions.py` - Visualization utilities
- `optimization_loglikelihood/` - MLE framework with Fisher information
