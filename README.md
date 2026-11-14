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

## Repository Structure

### `models/`
Pre-trained model weights (.h5) for MNIST (LeNet-1, LeNet-5), Fashion-MNIST (LeNet-5, ResNet-20), and CIFAR-10 (ResNet-20). Each model is trained with a different random seed for ensemble diversity.

### `model/`
Pre-trained SVHN LeNet-5 model weights (.h5), 5 seeds.

### `retrained_models/`
Models retrained using guidance-based test case selection on Fashion-MNIST. Each seed folder contains models retrained on the top-k prioritized test cases per metric.

### `retrainedmnist_models/`
Models retrained using guidance-based test case selection on MNIST. Each seed folder contains models retrained on the top-k prioritized test cases per metric, along with `guidance_results.csv` summarizing retraining outcomes.

### `code/`
- `notebooks/` — End-to-end experiment scripts per dataset (cifar10, fashion-mnist, mnist, svhn)
- `guidance/` — Guidance-based retraining scripts
- `signifancetest.py` — Table 3 significance tests (paired bootstrap + Holm-Bonferroni)
- `stattestfashion10model.py` — 10-model statistical tests (permutation, Wilcoxon, Cohen's d_z)
- `transferbitycheck.py` — Exclusive vs Inclusive APFD comparison across all datasets

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
