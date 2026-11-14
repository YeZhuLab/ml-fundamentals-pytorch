# Day 2: Logistic Regression and Binary Classification

## Objective

Implement Logistic Regression from scratch using NumPy and understand binary classification, optimization, and model evaluation.

## Implementation

- Synthetic binary classification dataset
- Sigmoid activation function
- Forward propagation
- Binary Cross-Entropy loss
- Analytical gradient computation
- Gradient Descent optimization
- Decision boundary visualization
- Confusion matrix
- Accuracy, Precision, Recall, and F1-score
- Learning rate experiments
- Classification threshold experiments

## Experiment 1: Learning Rate Comparison

Compared four learning rates over 1,000 epochs.

| Learning Rate | Final Loss |
|---|---:|
| 0.001 | 0.543688 |
| 0.01 | 0.327432 |
| 0.1 | 0.070768 |
| 1.0 | 0.015541 |

Observation:
A larger learning rate accelerated convergence in this experiment. However, LR = 1.0 showed initial instability.

## Experiment 2: Classification Metrics

At threshold = 0.5:

| Metric | Score |
|---|---:|
| Accuracy | 93.00% |
| Precision | 87.72% |
| Recall | 100.00% |
| F1-score | 93.46% |

## Experiment 3: Classification Threshold

Compared thresholds of 0.2, 0.5, and 0.8.

Observation:
Increasing the threshold increased Precision and reduced Recall in this experiment.

## Key Takeaways

- Understand sigmoid and probability-based classification.
- Derive and implement Logistic Regression gradients.
- Understand the impact of learning rates on optimization.
- Interpret confusion matrices and classification metrics.
- Understand the Precision-Recall trade-off.
- Understand how classification thresholds affect model predictions.

## Technologies

Python, NumPy, Matplotlib