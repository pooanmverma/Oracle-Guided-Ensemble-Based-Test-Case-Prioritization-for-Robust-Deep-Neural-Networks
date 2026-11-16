# Oracle-Guided Ensemble-Based Test Case Prioritization for Robust Deep Neural Networks



## Overview

Test case prioritization (TCP) for deep neural networks aims to reorder a test pool so that fault-revealing inputs are executed early, reducing the cost of quality assurance. Existing uncertainty-based TCP methods rely on a single model's output, missing the richer signal available from an ensemble.

We propose a **rotating oracle-guided deep ensemble framework**: a pool of *N* independently trained models takes turns as the *oracle* (the model under test), while the remaining models (or all models, depending on the setting) form the ensemble that computes uncertainty scores. Each test input is ranked by its estimated likelihood of exposing a fault in the current oracle.

### Key Contributions

- A systematic comparison of **9 uncertainty metrics** (4 single-model, 5 ensemble-based) and **coverage-based baselines** across 6 dataset–architecture pairs.
- **AggregatedMI**, a new ensemble metric defined as `0.5 × normalize(entropy) + 0.5 × normalize(mutual_info)`, which consistently ranks among the top performers.
- Two ensemble configurations — **Exclusive** (oracle held out) and **Inclusive** (oracle included) — with statistical evidence that Inclusive matching or outperforming Exclusive.
- **Guidance-based retraining**: using prioritization rankings to guide targeted model retraining and measuring downstream accuracy improvement.
- Rigorous statistical validation: paired bootstrap, Holm-Bonferroni correction, exact permutation tests, Wilcoxon signed-rank, and Cohen's d_z effect sizes.

## Uncertainty Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Single-model | DeepGini | 1 − Σpᵢ² |
| Single-model | Entropy | −Σpᵢ log pᵢ |
| Single-model | Softmax | max(p) — lower = more uncertain |
| Single-model | PCS | gap between top-2 class probabilities |
| Ensemble | EnsGini | DeepGini on mean ensemble prediction |
| Ensemble | Variance | mean per-class variance across ensemble members |
| Ensemble | Disagreement | 1 − max(mean prediction) |
| Ensemble | MutualInfo | ensemble entropy minus mean member entropy |
| Ensemble | **AggregatedMI** | 0.5 × norm(entropy) + 0.5 × norm(mutual_info) |

## Ensemble Settings

| Setting | Oracle in ensemble? | Ensemble size (N models total) |
|---------|-------------------|-------------------------------|
| **Exclusive** | No | N − 1 |
| **Inclusive** | Yes | N |

## Datasets and Models

| Dataset | Model |  # Ensemble Members 
|---------|-------|--------------------
| MNIST | LeNet-1,LeNet-5, |          5 
|Fashion-MNIST |Lenet1,Lenet5   |          5 (3,7 10 for ablation) 
|SVHN | LeNet-5   |          5 
| CIFAR-10        | Resnet20  |       5 

Each ensemble member is trained with a different random seed to ensure diversity.

### Dataset Sources

- **MNIST**: `tf.keras.datasets.mnist` (auto-download)
- **Fashion-MNIST**: `tf.keras.datasets.fashion_mnist` (auto-download)
- **CIFAR-10**: `tf.keras.datasets.cifar10` (auto-download)
- **SVHN**: Download `test_32x32.mat` from [SVHN dataset page](http://ufldl.stanford.edu/housenumbers/) → place in `code/data/`

## Adversarial Attacks

Test pools include both clean and adversarial inputs. Adversarial examples are generated using:

| Attack | Description |
|--------|-------------|
| **FGSM** | Fast Gradient Sign Method — single-step perturbation |
| **BIM** | Basic Iterative Method — multi-step iterative FGSM |

Adversarial `.npy` files are stored in `adv_data*/` directories, named as `{dataset}_{attack}_{model}.npy`.

## Repository Structure

```
├── code/
│   ├── notebooks/
│   │   ├── cifar10/                # CIFAR-10 ResNet-20 experiments
│   │   ├── fashion-mnist/          # Fashion-MNIST experiments (LeNet-5, ResNet-20, ablation)
│   │   ├── mnist/                  # MNIST experiments (LeNet-1, LeNet-5)
│   │   └── svhn/                   # SVHN LeNet-5 experiments
│   ├── guidance/                   # Guidance-based retraining scripts
│   │   ├── guideranking.py         # Ranking-guided retraining pipeline
│   │   └── guidanceallfashion.py   # Fashion-MNIST guidance experiments
│   ├── signifancetest.py           # Table 3: bootstrap + Holm-Bonferroni significance tests
│   ├── stattestfashion10model.py   # 10-model ablation: permutation test, Wilcoxon, Cohen's d_z
│   └── transferbitycheck.py        # Exclusive vs Inclusive APFD comparison
├── models/                         # Trained model weights (.h5) — CIFAR-10, Fashion-MNIST, MNIST
├── model/                          # SVHN model weights (.h5)
├── retrained_models/               # Retrained models from guidance (Fashion-MNIST)
├── retrainedmnist_models/          # Retrained models from guidance (MNIST)
├── guidance_results/               # Guidance retraining results (CSVs)
├── results/                        # APFD results per experiment (JSON)
├── plotrq3.py                      # RQ3 visualization script
└── README.md
```

## Statistical Analysis

### Table 5 Significance Tests (`signifancetest.py`)
- Paired bootstrap test (B = 10,000) comparing best ensemble metric vs best single-model metric per dataset × setting cell
- Holm-Bonferroni correction across all cells for family-wise error control

### 10-Model Ablation Study (`stattestfashion10model.py`)
- Fashion-MNIST with 10 ensemble members
- Exact permutation test (2¹⁰ = 1,024 permutations)
- Wilcoxon signed-rank test
- Cohen's d_z effect size

### Exclusive vs Inclusive (`transferbitycheck.py`)
- Per-attack (FGSM, BIM) APFD comparison across all 6 dataset–model pairs
- Reports percentage-point gain of Inclusive over Exclusive

## Requirements

```
Python >= 3.8
TensorFlow >= 2.x
NumPy
SciPy
```

## Usage

### Running Experiments

Each notebook in `code/notebooks/` runs the full TCP pipeline for its dataset:

```bash
cd code/notebooks/mnist
python MNISTFOURSETTINGSlenet1final.py      # MNIST LeNet-1
python MNISTFOURSETTINGSlenet5final.py      # MNIST LeNet-5

cd ../fashion-mnist
python fashion_coverage_comparisonfashionmnist_lenent5end_to_end.py   # Fashion-MNIST LeNet-5
python FSAHIONMNISTFOURSETTINGSrenet20final.py                       # Fashion-MNIST ResNet-20
python fashion3modelablationstudy.py                                   # 3-model ablation

cd ../cifar10
python cifar10_resnet20_four_settings.py     # CIFAR-10 ResNet-20

cd ../svhn
python SVHNFOURSETTINGSlenet5final.py       # SVHN LeNet-5
```

Statistical Analysis
5-Model Significance (signifancetest.py)
Exact two-sided paired permutation test across 5 rotating-oracle rounds per comparison
With n=5, minimum attainable p-value is 2/2⁵ = 0.0625 (cannot reach α=0.05 per-comparison)

Median paired Cohen's d_z = 7.51 (large standardized effect)
Paired bootstrap (B=10,000) with one Holm-Bonferroni correction across all cells
cd code

python signifancetest.py           # Table 3 significance tests
python stattestfashion10model.py   # 10-model ablation statistics
python transferbitycheck.py        # Exclusive vs Inclusive comparison
```
10-Model Ablation Study (stattestfashion10model.py)
Supplementary experiment: Fashion-MNIST with K=10 independently trained LeNet-5 models (n=10 oracle rounds)
Two-sided Wilcoxon signed-rank test rejects null for every ensemble metric under both FGSM and BIM (p=0.002 for all 10 comparisons, significant after Holm-Bonferroni at α=0.05)
Inclusive ensemble achieves strictly higher APFD in all 100 pairwise comparisons (10 oracles × 5 metrics × 2 attacks)
Median Cohen's d_z = 5.23
Exact permutation test (2¹⁰ = 1,024 permutations, minimum p ≈ 0.002)


### Guidance-Based Retraining

```bash
cd code/guidance
MNISTtrain_guideance              #  mnist Guidance retraining 
python guideranking.py             # mnist Guidance retraining 
python guideance_mnist             #mnist Guidance retraining 
python guidanceallfashion.py       # Fashion-MNIST guidance
coveragealllsvhnretrain             # SVHN guidance
```

## Evaluation Metric

**APFD (Average Percentage of Faults Detected)**: measures how early faults are found in the prioritized test sequence. Higher APFD = faults detected earlier = better prioritization.

```
APFD = 1 − (Σ rank of faulty inputs) / (n × m) + 1/(2n)
```

where *n* = total test inputs, *m* = number of faulty inputs.

## License

This project is released for research purposes.

## Citation

Citation details will be added upon publication.