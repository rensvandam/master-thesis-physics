# Optimization / Log-Likelihood

Maximum likelihood estimation (MLE) framework for emitter localization.

## Files

### `backward_model.py`
The inverse problem solver for estimating emitter positions from photon measurements.

- `Fitter` class for MLE parameter estimation
- Expected photon count calculations per pixel
- Log-likelihood function implementation
- Fisher information matrix for uncertainty quantification
- Cramer-Rao bound analysis

### `derivatives.py`
Analytical gradients for optimization algorithms.

- Gradient and Hessian of the log-likelihood function
- Derivatives of expected photon counts w.r.t. position and intensity
- Error function-based PSF integration over pixel areas
- Support for multi-parameter fitting

## Usage

```python
from project.model.optimization_loglikelihood.backward_model import Fitter

fitter = Fitter(sensor=sensor, psf_sigma=0.1)
estimated_position = fitter.fit(photon_counts)
uncertainty = fitter.cramer_rao_bound()
```

## Theory

The MLE approach maximizes the likelihood of observing the measured photon counts given the model parameters (emitter position, intensity). The Fisher information matrix provides theoretical bounds on localization precision.
