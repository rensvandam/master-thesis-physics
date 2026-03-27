# Project

This is the main codebase for the SPAD-SMLM simulation framework.

## Directory Overview

| Folder | Description |
|--------|-------------|
| `model/` | Core simulation modules (emitters, sensors, localization, coherence) |
| `simulations/` | Experiment scripts and examples |
| `test/` | Unit tests for all modules |
| `data/` | Configuration files (PSF parameters) |
| `figures/` | Historical meeting figures |
| `report/` | Jupyter notebooks used for thesis figures |

## Quick Navigation

- **New users**: Start with `simulations/example.py`
- **Understanding the code**: Read `model/README.md`
- **Running tests**: `pytest test/`

## Module Dependencies

```
sample.py          Fluorescent emitter models
    ↓
detection.py       SPAD sensor simulation
    ↓
setup.py           Experiment configuration
    ↓
localization.py    Position estimation
coherence_from_data.py  Photon statistics
ISMprocessor.py    Resolution enhancement
```
