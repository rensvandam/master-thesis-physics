# Simulations

Simulation scripts and experiments for the SPAD-SMLM framework.

## Directory Structure

```
simulations/
├── examples/       # Start here - basic usage examples
├── evaluation/     # Core localization evaluation scripts
├── experiments/    # Parameter studies and analysis
├── notebooks/      # Jupyter notebooks for visualization
└── deprecated/     # Old/unused scripts
```

## Getting Started

New users should start with the examples:

```bash
cd examples/
python example.py
```

## Folders

### `examples/`
Entry-point scripts demonstrating basic framework usage:
- `example.py` - Basic SPAD23 scanning simulation
- `run_scanning_experiment.py` - Configurable scanning experiment
- `widefield_example_sim.py` - Widefield (non-scanning) mode

### `evaluation/`
Core thesis evaluation scripts for localization performance:
- `compute_localization_bias_precision.py` - Main bias/precision analysis
- `evaluate_localization.py` - Single-run evaluation
- `evaluate_localization_parallel.py` - Parallel evaluation runs

### `experiments/`
Parameter studies and specific analyses:
- `variable_deadtime.py` - Dead time parameter sweep
- `variable_emitter_*.py` - Emitter property studies
- `dead_time_*.py` - Dead time analysis
- `coherence_test.py` - Coherence validation

### `notebooks/`
Jupyter notebooks for visualization and reporting:
- `quick_visual.ipynb` - Quick result visualization
- `report_plots.ipynb` - Thesis figure generation
