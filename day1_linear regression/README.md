# Day 1: Linear Regression and Gradient Descent

## Objectives
- Implement Linear Regression using NumPy.
- Understand Forward Propagation.
- Implement Mean Squared Error (MSE).
- Calculate gradients manually.
- Implement Gradient Descent.
- Compare different learning rates.

## Implementation
- Synthetic dataset generation
- Forward propagation
- Loss calculation
- Gradient computation
- Parameter updates
- Prediction visualization

## Experiments

Compared learning rates:

`0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0`

### Findings
- Small learning rates result in slower convergence.
- Appropriate learning rates provide faster convergence.
- Excessively large learning rates can cause divergence.

## Technologies
- Python
- NumPy
- Matplotlib

## Run

```bash
python linear_regression.py
python learning_rate_analysis.py
```