# Evaluation

Core scripts for evaluating localization performance.

## Files

### `compute_localization_bias_precision.py`
**Main evaluation script.** Comprehensive bias and precision analysis for localization algorithms. This is the primary script used for thesis results.

### `evaluate_localization.py`
Single-run localization evaluation with distance metrics computation.

Key functions:
- `run_evaluation()` - Run complete evaluation
- `compute_distance_metrics()` - Calculate localization errors

### `evaluate_localization_algorithm.py`
Algorithm-specific evaluation and comparison utilities.

### `evaluate_localization_parallel.py`
Parallel version of localization evaluation for running many simulations efficiently.

```python
from project.simulations.evaluation.evaluate_localization_parallel import run_evaluation_parallel

results = run_evaluation_parallel(
    n_runs=100,
    emitter_densities=[1, 5, 10],
    # ... parameters
)
```
