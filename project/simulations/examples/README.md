# Examples

Basic usage examples to get started with the SPAD-SMLM framework.

## Files

### `example.py`
**Start here.** Basic scanning microscopy simulation with SPAD23 sensor.

```bash
python example.py
```

### `run_scanning_experiment.py`
Configurable scanning experiment with full parameter control. Used by other scripts.

```python
from project.simulations.examples.run_scanning_experiment import run_scanning_experiment

setup, photon_map, G2_map, n_emitters_map, metadata = run_scanning_experiment(
    emitter_density=5,
    laser_power=50e3,
    dwell_time=1e-3,
    # ... more parameters
)
```

### `widefield_example_sim.py`
Example using widefield illumination mode with the larger SPAD512 sensor.
