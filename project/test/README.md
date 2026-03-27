# Tests

Unit tests for the SPAD-SMLM framework using pytest.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test_sample.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=model
```

## Test Files

| File | Tests |
|------|-------|
| `test_sample.py` | Fluorescent emitter models, photon emission |
| `test_detection.py` | SPAD sensor behavior, dead time effects |
| `test_coherence.py` | Coherence calculations, g2 functions |
| `test_helper_functions.py` | Utility functions, array operations |
| `test_emitter_density_map.py` | Spatial density estimation |
| `test_backward_model.py` | MLE fitting, likelihood functions |
| `test_derivatives.py` | Gradient calculations for optimization |

## Writing New Tests

Tests follow the standard pytest conventions:

```python
def test_emitter_creation():
    emitter = Alexa647(position=(0, 0))
    assert emitter.position == (0, 0)
```
