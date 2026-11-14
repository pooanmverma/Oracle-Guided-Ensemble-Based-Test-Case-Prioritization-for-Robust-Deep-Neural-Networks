# Oracle-Guided Ensemble-Based Test Case Prioritization for Robust Deep Neural Networks

This repository contains the code, trained models, and results for test case prioritization (TCP) for deep neural networks using a rotating oracle-guided deep ensemble framework.

## Overview

A framework that leverages deep ensembles with a rotating oracle strategy to prioritize test cases for DNNs. Uncertainty metrics computed from an ensemble of models — where each model takes a turn as the "oracle" (the model under test) — rank test inputs by their likelihood of revealing faults.

### Uncertainty Metrics

| Type | Metrics |
|------|---------|
| **Single-model** | DeepGini, Entropy, Softmax, PCS |
| **Ensemble** | EnsGini, Variance, Disagreement, MutualInfo, AdaptiveMI |

### Ensemble Settings

- **Exclusive**: Oracle model is held out of the ensemble
- **Inclusive**: Oracle model is included in the ensemble

## Datasets and Models

| Dataset | Model | # Seeds |
|---------|-------|---------|
| MNIST | LeNet-1 | 5 |
| MNIST | LeNet-5 | 5 |
| Fashion-MNIST | LeNet-5 | 5 (+ 10 for ablation) |
| Fashion-MNIST | ResNet-20 | 5 |
| SVHN | LeNet-5 | 5 |
| CIFAR-10 | ResNet-20 | 5 |

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy, SciPy

## Usage

```bash
cd code
python signifancetest.py
python stattestfashion10model.py
python transferbitycheck.py
```
