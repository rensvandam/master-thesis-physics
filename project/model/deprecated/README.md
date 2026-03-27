# Deprecated Model Files

These files are no longer actively used in the main codebase but are preserved for reference.

## Files

| File | Original Purpose | Reason Deprecated |
|------|------------------|-------------------|
| `localization_speed.py` | Numba-optimized localization | Never integrated into main workflow |
| `coherence_analytical.py` | Analytical coherence expressions | Only used by deprecated simulations |
| `emitter_density_map.py` | Spatial density estimation | Only used by deprecated simulations |
| `plot_functions.py` | Visualization utilities | Only used by dashboard (deprecated) |

## Folders

### `optimization_loglikelihood/`
Maximum likelihood estimation framework with Fisher information and Cramer-Rao bounds. Only used in deprecated simulations and tests.

- `backward_model.py` - MLE fitting class
- `derivatives.py` - Analytical gradients

## Dashboard Tools (from project/tool/)

The following dashboard files were also moved to `project/simulations/deprecated/`:
- `dashboard.py` - Interactive Dash application
- `dashboard_new.py` - Updated dashboard version
- `dash_plot_functions.py` - Dashboard plotting utilities

## Note

If you need functionality from these files, consider integrating the relevant parts into the active modules rather than importing directly from deprecated code.
