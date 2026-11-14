#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fashion-MNIST: MC-Dropout (1 model, T=5) vs Inclusive Deep Ensemble (5 models)

MC-Dropout:
    1 trained dropout model
        -> T=5 stochastic forward passes
        -> mean predictive probability
        -> uncertainty score
        -> ranking
        -> APFD

Deep Ensemble:
    5 independently trained models
        -> one deterministic prediction/model
        -> mean ensemble probability
        -> uncertainty score
        -> ranking
        -> APFD for each rotating oracle
        -> mean ± std across 5 oracle runs

Dataset:
    Clean Fashion-MNIST test set only.

Compared measures:
    EnsGini / MC-Gini
    Class-wise Variance
    Disagreement
    MI
    Aggregated MI

Important:
    The MC-Dropout result is ONE APFD value per metric because the five
    stochastic passes are first averaged into one predictive distribution.
    The std reported for the Deep Ensemble is across its five rotating
    oracle runs. A std across MC-Dropout passes is NOT reported because
    the passes are not separately ranked/evaluated in this protocol.
"""

# =====================================================
# THREAD CAPS BEFORE NUMPY / TENSORFLOW
# =====================================================
import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

# =====================================================
# IMPORTS
# =====================================================
import time
import json
import random
import numpy as np
import tensorflow as tf

# =====================================================
# CONFIG
# =====================================================
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# One trained MC-Dropout model
MC_DROPOUT_MODEL = f"{BASE}/models/fashion_drop_seed0.h5"

# Five independently trained Deep Ensemble models
ENSEMBLE_SEEDS = [0, 1, 2, 3, 4]

ENSEMBLE_MODELS = {
    f"seed{s}": f"{BASE}/models/fashion_seed{s}.h5"
    for s in ENSEMBLE_SEEDS
}

# MC-Dropout stochastic forward passes
T = 50

BATCH = 256

# Reproducible tie-breaking / random ordering
SEED = 42
rng = np.random.default_rng(SEED)

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================================
# HELPERS
# =====================================================
def deepgini(p):
    return 1.0 - np.sum(p ** 2, axis=1)


def entropy(p):
    return -np.sum(
        p * np.log(p + 1e-12),
        axis=1
    )


def classwise_variance(predictions):
    """
    Mean variance across classes.

    predictions shape:
        (number_of_models_or_passes, N, num_classes)
    """
    return np.mean(
        np.var(predictions, axis=0),
        axis=1
    )


def disagreement(mean_probs):
    """
    Ensemble-level aggregated uncertainty used in the
    existing implementation:
        1 - max(mean predictive probability)
    """
    return 1.0 - np.max(
        mean_probs,
        axis=1
    )


def mutual_information(predictions):
    """
    MI = H(mean prediction) - mean(H(individual prediction)).
    """
    mean_probs = np.mean(
        predictions,
        axis=0
    )

    predictive_entropy = -np.sum(
        mean_probs * np.log(mean_probs + 1e-12),
        axis=1
    )

    individual_entropy = -np.sum(
        predictions * np.log(predictions + 1e-12),
        axis=2
    )

    return (
        predictive_entropy
        - np.mean(individual_entropy, axis=0)
    )


def normalize(x):
    return (
        (x - x.min())
        / (
            x.max() - x.min() + 1e-12
        )
    )


def aggregated_mi(mean_probs, mi):
    """
    Aggregated MI:
        0.5 * normalized predictive entropy
        + 0.5 * normalized MI
    """
    return (
        0.5 * normalize(entropy(mean_probs))
        + 0.5 * normalize(mi)
    )


def rank_desc(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return np.lexsort(
        (
            rng.random(len(scores)),
            -scores
        )
    )


def APFD(rank, faulty):
    """
    Average Percentage of Faults Detected.
    """
    n = len(faulty)
    m = int(faulty.sum())

    if m == 0:
        return 0.0

    positions = np.empty(
        n,
        dtype=np.int64
    )

    positions[np.asarray(rank)] = np.arange(
        1,
        n + 1
    )

    return (
        1.0
        - positions[faulty.astype(bool)].sum()
        / (n * m)
        + 1.0 / (2.0 * n)
    )



# =====================================================
# DATA / ADVERSARIAL CONFIGURATION
# =====================================================
DATA_SETTINGS = ["clean", "adversarial"]
ENSEMBLE_SETTINGS = ["exclusive", "inclusive"]

FASHION_ADV_DIR = f"{BASE}/adv_data_fashion"

# IMPORTANT:
# The MC-Dropout model needs adversarial examples generated
# specifically against the SAME dropout model. Do not silently
# reuse adversarial examples generated against a different model.
#
# Expected files for the one MC-Dropout model:
#   fashion_drop_fgsm_seed0.npy
#   fashion_drop_bim_seed0.npy
#
# Expected ensemble files:
#   fashion_fgsm_seed0.npy ... fashion_fgsm_seed4.npy
#   fashion_bim_seed0.npy  ... fashion_bim_seed4.npy
#
# Change these names ONLY if your existing files use another
# confirmed naming convention.
MC_DROPOUT_ADV_FGSM = os.path.join(
    FASHION_ADV_DIR,
    "fashion_drop_fgsm_seed0.npy"
)
MC_DROPOUT_ADV_BIM = os.path.join(
    FASHION_ADV_DIR,
    "fashion_drop_bim_seed0.npy"
)

# =====================================================
# DATA
# =====================================================
def load_fashion():
    (_, _), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    x_test = (
        x_test.astype("float32") / 255.0
    )

    x_test = np.expand_dims(
        x_test,
        -1
    )

    y_test = y_test.astype("int64")

    return x_test, y_test


# =====================================================
# ADVERSARIAL DATA
# =====================================================
def load_dropout_adv():
    """
    Load adversarial examples generated specifically for
    the one MC-Dropout model.
    """
    for path in (
        MC_DROPOUT_ADV_FGSM,
        MC_DROPOUT_ADV_BIM,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "\nMissing MC-Dropout adversarial file:\n"
                f"{path}\n\n"
                "The adversarial examples must be generated against "
                "the same fashion_drop_seed0.h5 model used for "
                "MC-Dropout evaluation."
            )

    x_fgsm = np.load(
        MC_DROPOUT_ADV_FGSM
    ).astype("float32")

    x_bim = np.load(
        MC_DROPOUT_ADV_BIM
    ).astype("float32")

    return x_fgsm, x_bim


def load_ensemble_adv(seed):
    """
    Load model-specific FGSM and BIM adversarial examples
    for the current rotating oracle.
    """
    fgsm_path = os.path.join(
        FASHION_ADV_DIR,
        f"fashion_fgsm_seed{seed}.npy"
    )
    bim_path = os.path.join(
        FASHION_ADV_DIR,
        f"fashion_bim_seed{seed}.npy"
    )

    if not os.path.isfile(fgsm_path):
        raise FileNotFoundError(
            f"Missing ensemble FGSM adversarial file:\n{fgsm_path}"
        )

    if not os.path.isfile(bim_path):
        raise FileNotFoundError(
            f"Missing ensemble BIM adversarial file:\n{bim_path}"
        )

    x_fgsm = np.load(
        fgsm_path
    ).astype("float32")

    x_bim = np.load(
        bim_path
    ).astype("float32")

    return x_fgsm, x_bim


# =====================================================
# MODEL VERIFICATION
# =====================================================
def verify_models():
    print("\nChecking MC-Dropout model...")
    if not os.path.isfile(MC_DROPOUT_MODEL):
        raise FileNotFoundError(
            f"Missing MC-Dropout model:\n{MC_DROPOUT_MODEL}"
        )

    print(f"MC-Dropout: {MC_DROPOUT_MODEL}")

    print("\nChecking 5 ensemble models...")
    missing = []

    for name, path in ENSEMBLE_MODELS.items():
        print(f"{name}: {path}")
        if not os.path.isfile(path):
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            "\nMissing ensemble model file(s):\n"
            + "\n".join(missing)
        )

    real_paths = [
        os.path.realpath(p)
        for p in ENSEMBLE_MODELS.values()
    ]

    if len(set(real_paths)) != len(real_paths):
        raise RuntimeError(
            "Duplicate ensemble model paths detected."
        )

    print("Verified 5 unique ensemble model files.")


# =====================================================
# MC-DROPOUT
# =====================================================
def evaluate_mc_dropout(
    model,
    x_test,
    y_test,
    data_setting
):
    print("\n" + "=" * 70)
    print(
        f"MC-DROPOUT | 1 MODEL | T={T} | "
        f"{data_setting.upper()} DATA"
    )
    print("=" * 70)

    # -------------------------------------------------
    # Build the MC-Dropout evaluation pool
    # -------------------------------------------------
    if data_setting == "clean":

        x_all = x_test
        y_all = y_test

        print(
            f"Evaluation pool: {len(x_all)} clean samples"
        )

    elif data_setting == "adversarial":

        x_fgsm, x_bim = load_dropout_adv()

        if len(x_fgsm) != len(x_test):
            raise ValueError(
                "MC-Dropout FGSM size does not match "
                "Fashion-MNIST test-set size."
            )

        if len(x_bim) != len(x_test):
            raise ValueError(
                "MC-Dropout BIM size does not match "
                "Fashion-MNIST test-set size."
            )

        x_all = np.concatenate(
            [
                x_test,
                x_fgsm,
                x_bim
            ],
            axis=0
        )

        y_all = np.tile(
            y_test,
            3
        )

        print(
            f"Evaluation pool: {len(x_all)} "
            f"(clean={len(x_test)}, "
            f"FGSM={len(x_fgsm)}, "
            f"BIM={len(x_bim)})"
        )

    else:
        raise ValueError(
            "data_setting must be 'clean' or 'adversarial'."
        )

    # -------------------------------------------------
    # Deterministic prediction for fault identification
    # -------------------------------------------------
    deterministic_probs = model.predict(
        x_all,
        batch_size=BATCH,
        verbose=0
    )

    deterministic_pred = np.argmax(
        deterministic_probs,
        axis=1
    )

    faulty = (
        deterministic_pred != y_all
    ).astype(int)

    print(
        f"Faulty inputs: {int(faulty.sum())}"
    )

    # -------------------------------------------------
    # T=5 stochastic forward passes
    # -------------------------------------------------
    stochastic_predictions = []

    start_prediction = time.time()

    for t in range(T):

        probs = model(
            x_all,
            training=True
        ).numpy()

        stochastic_predictions.append(
            probs
        )

        print(
            f"Stochastic pass {t + 1}/{T}"
        )

    prediction_time = (
        time.time() - start_prediction
    )

    predictions = np.stack(
        stochastic_predictions,
        axis=0
    )

    print(
        f"Stochastic prediction shape: "
        f"{predictions.shape}"
    )

    # -------------------------------------------------
    # Mean predictive probability
    # -------------------------------------------------
    mean_probs = np.mean(
        predictions,
        axis=0
    )

    # -------------------------------------------------
    # Scores
    # -------------------------------------------------
    mi = mutual_information(
        predictions
    )

    scores = {
        "MC-Gini": deepgini(mean_probs),
        "Class-wise Variance": classwise_variance(predictions),
        "Disagreement": disagreement(mean_probs),
        "MI": mi,
        "Aggregated MI": aggregated_mi(
            mean_probs,
            mi
        ),
    }

    # -------------------------------------------------
    # Ranking + APFD
    # -------------------------------------------------
    apfd = {}

    print(
        f"\n------ MC-DROPOUT APFD | "
        f"{data_setting.upper()} ------"
    )

    for metric, score in scores.items():

        rank = rank_desc(score)

        apfd[metric] = APFD(
            rank,
            faulty
        )

        print(
            f"{metric:22s}: "
            f"{apfd[metric]:.4f}"
        )

    print(
        f"\nMC-Dropout prediction time "
        f"(T={T}): {prediction_time:.4f} s"
    )

    return {
        "apfd": apfd,
        "prediction_time": prediction_time,
        "num_samples": len(x_all),
        "num_faulty": int(faulty.sum()),
    }


# =====================================================
# DEEP ENSEMBLE
# =====================================================
def evaluate_deep_ensemble(
    models,
    x_test,
    y_test,
    data_setting,
    ensemble_setting
):
    print("\n" + "=" * 70)
    print(
        f"DEEP ENSEMBLE | 5 MODELS | "
        f"{ensemble_setting.upper()} | "
        f"{data_setting.upper()} DATA"
    )
    print("=" * 70)

    names = list(
        models.keys()
    )

    ensemble_apfd_by_oracle = {
        oracle: {}
        for oracle in names
    }

    ensemble_prediction_time_by_oracle = {}

    # -------------------------------------------------
    # Rotating oracle
    # -------------------------------------------------
    for oracle in names:

        print("\n" + "-" * 70)
        print(
            f"ORACLE = {oracle} | "
            f"{ensemble_setting.upper()} | "
            f"{data_setting.upper()}"
        )
        print("-" * 70)

        # -------------------------------------------------
        # Build oracle-specific evaluation pool
        # -------------------------------------------------
        if data_setting == "clean":

            x_all = x_test
            y_all = y_test

        elif data_setting == "adversarial":

            oracle_seed = int(
                oracle.replace("seed", "")
            )

            x_fgsm, x_bim = load_ensemble_adv(
                oracle_seed
            )

            if len(x_fgsm) != len(x_test):
                raise ValueError(
                    f"FGSM size mismatch for {oracle}: "
                    f"{len(x_fgsm)} vs {len(x_test)}"
                )

            if len(x_bim) != len(x_test):
                raise ValueError(
                    f"BIM size mismatch for {oracle}: "
                    f"{len(x_bim)} vs {len(x_test)}"
                )

            x_all = np.concatenate(
                [
                    x_test,
                    x_fgsm,
                    x_bim
                ],
                axis=0
            )

            y_all = np.tile(
                y_test,
                3
            )

        else:
            raise ValueError(
                "data_setting must be 'clean' or 'adversarial'."
            )

        # -------------------------------------------------
        # Predictions from all five models
        # -------------------------------------------------
        prediction_start = time.time()

        all_probs = np.stack(
            [
                models[name].predict(
                    x_all,
                    batch_size=BATCH,
                    verbose=0
                )
                for name in names
            ],
            axis=0
        )

        prediction_time = (
            time.time() - prediction_start
        )

        ensemble_prediction_time_by_oracle[
            oracle
        ] = prediction_time

        print(
            f"Prediction shape: {all_probs.shape}"
        )

        # -------------------------------------------------
        # Oracle and fault labels
        # -------------------------------------------------
        oracle_idx = names.index(
            oracle
        )

        oracle_probs = all_probs[
            oracle_idx
        ]

        oracle_pred = np.argmax(
            oracle_probs,
            axis=1
        )

        faulty = (
            oracle_pred != y_all
        ).astype(int)

        print(
            f"Evaluation pool: {len(x_all)} | "
            f"Faulty inputs: {int(faulty.sum())}"
        )

        # -------------------------------------------------
        # Exclusive / inclusive ensemble
        # -------------------------------------------------
        if ensemble_setting == "exclusive":

            ensemble_mask = (
                np.arange(
                    len(all_probs)
                )
                != oracle_idx
            )

            ens = all_probs[
                ensemble_mask
            ]

        elif ensemble_setting == "inclusive":

            ens = all_probs

        else:
            raise ValueError(
                "ensemble_setting must be 'exclusive' or 'inclusive'."
            )

        expected_size = (
            len(names) - 1
            if ensemble_setting == "exclusive"
            else len(names)
        )

        assert ens.shape[0] == expected_size, (
            f"Expected {expected_size} ensemble models, "
            f"got {ens.shape[0]}"
        )

        print(
            f"Ensemble shape: {ens.shape}"
        )

        # -------------------------------------------------
        # Mean ensemble prediction
        # -------------------------------------------------
        mean_probs = np.mean(
            ens,
            axis=0
        )

        # -------------------------------------------------
        # Ensemble uncertainty measures
        # -------------------------------------------------
        ens_gini = deepgini(
            mean_probs
        )

        variance = classwise_variance(
            ens
        )

        dis = disagreement(
            mean_probs
        )

        mi = mutual_information(
            ens
        )

        ami = aggregated_mi(
            mean_probs,
            mi
        )

        scores = {
            "EnsGini": ens_gini,
            "Class-wise Variance": variance,
            "Disagreement": dis,
            "MI": mi,
            "Aggregated MI": ami,
        }

        # -------------------------------------------------
        # Ranking + APFD
        # -------------------------------------------------
        for metric, score in scores.items():

            rank = rank_desc(
                score
            )

            ensemble_apfd_by_oracle[
                oracle
            ][metric] = APFD(
                rank,
                faulty
            )

        print(
            "\n------ APFD FOR CURRENT ORACLE ------"
        )

        for metric, value in (
            ensemble_apfd_by_oracle[
                oracle
            ].items()
        ):
            print(
                f"{metric:22s}: {value:.4f}"
            )

    # -------------------------------------------------
    # Mean ± std across the five rotating-oracle runs
    # -------------------------------------------------
    ensemble_mean_std = {}

    print("\n" + "=" * 70)
    print(
        f"DEEP ENSEMBLE FINAL RESULTS | "
        f"{data_setting.upper()} | "
        f"{ensemble_setting.upper()} | MEAN ± STD"
    )
    print("=" * 70)

    metrics = [
        "EnsGini",
        "Class-wise Variance",
        "Disagreement",
        "MI",
        "Aggregated MI",
    ]

    for metric in metrics:

        values = np.array(
            [
                ensemble_apfd_by_oracle[
                    oracle
                ][metric]
                for oracle in names
            ],
            dtype=np.float64
        )

        ensemble_mean_std[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.tolist(),
        }

        print(
            f"{metric:22s}: "
            f"{values.mean():.4f} ± "
            f"{values.std():.4f}"
        )

    return {
        "apfd_by_oracle": ensemble_apfd_by_oracle,
        "mean_std": ensemble_mean_std,
        "prediction_time_by_oracle": (
            ensemble_prediction_time_by_oracle
        ),
    }


# =====================================================
# COMPARISON TABLE
# =====================================================
def print_comparison(
    mc_results_by_setting,
    ensemble_results_by_setting
):
    print("\n" + "=" * 115)
    print(
        "MC-DROPOUT (1 MODEL, T=5) "
        "vs DEEP ENSEMBLE (5 MODELS)"
    )
    print("=" * 115)

    mapping = [
        ("EnsGini / MC-Gini", "MC-Gini", "EnsGini"),
        ("Class-wise Variance", "Class-wise Variance", "Class-wise Variance"),
        ("Disagreement", "Disagreement", "Disagreement"),
        ("MI", "MI", "MI"),
        ("Aggregated MI", "Aggregated MI", "Aggregated MI"),
    ]

    comparison = {}

    for data_setting in DATA_SETTINGS:

        for ensemble_setting in ENSEMBLE_SETTINGS:

            setting_key = (
                f"{data_setting}_{ensemble_setting}"
            )

            mc_results = (
                mc_results_by_setting[
                    data_setting
                ]
            )

            ensemble_results = (
                ensemble_results_by_setting[
                    setting_key
                ]
            )

            print("\n" + "-" * 115)
            print(
                f"SETTING: {data_setting.upper()} + "
                f"{ensemble_setting.upper()}"
            )
            print("-" * 115)

            print(
                f"{'Metric':24s}"
                f"{'MC-Dropout':22s}"
                f"{'Ensemble mean ± std':30s}"
                f"{'Difference':15s}"
            )

            print("-" * 115)

            comparison[setting_key] = {}

            for label, mc_metric, ens_metric in mapping:

                mc_value = float(
                    mc_results["apfd"][mc_metric]
                )

                ens_mean = float(
                    ensemble_results["mean_std"][
                        ens_metric
                    ]["mean"]
                )

                ens_std = float(
                    ensemble_results["mean_std"][
                        ens_metric
                    ]["std"]
                )

                difference = (
                    ens_mean - mc_value
                )

                comparison[
                    setting_key
                ][label] = {
                    "mc_dropout_apfd": mc_value,
                    "ensemble_mean": ens_mean,
                    "ensemble_std": ens_std,
                    "difference": difference,
                }

                print(
                    f"{label:24s}"
                    f"{mc_value:.4f}"
                    f"{'':16s}"
                    f"{ens_mean:.4f} ± {ens_std:.4f}"
                    f"{'':10s}"
                    f"{difference:+.4f}"
                )

    print("\n" + "=" * 115)

    print(
        "\nProtocol:"
    )
    print(
        "MC-Dropout = one trained Dropout model + T=5 stochastic "
        "passes, averaged before scoring and ranking."
    )
    print(
        "Deep Ensemble = five independently trained models; "
        "exclusive/inclusive is applied only to the ensemble."
    )
    print(
        "For the Deep Ensemble, mean ± std is computed across "
        "the five rotating-oracle APFD runs."
    )
    print(
        "For MC-Dropout, one APFD is produced per metric per data "
        "setting; no std across the five stochastic passes is reported."
    )

    return comparison


# =====================================================
# MAIN
# =====================================================
def main():

    total_start = time.time()

    print("\n" + "=" * 90)
    print(
        "FASHION-MNIST: MC-DROPOUT (1 MODEL, T=5) "
        "vs DEEP ENSEMBLE (5 MODELS)"
    )
    print("=" * 90)

    print(
        "\nFour comparison settings:"
    )
    print(
        "1. Clean + Exclusive"
    )
    print(
        "2. Clean + Inclusive"
    )
    print(
        "3. Adversarial + Exclusive"
    )
    print(
        "4. Adversarial + Inclusive"
    )

    # -------------------------------------------------
    # Verify model files
    # -------------------------------------------------
    verify_models()

    # Verify the one MC-Dropout model explicitly.
    if not os.path.isfile(MC_DROPOUT_MODEL):
        raise FileNotFoundError(
            f"Missing MC-Dropout model:\n{MC_DROPOUT_MODEL}"
        )

    # -------------------------------------------------
    # Load Fashion-MNIST clean test set
    # -------------------------------------------------
    x_test, y_test = load_fashion()

    print(
        f"\nClean test inputs: {len(x_test)}"
    )

    # -------------------------------------------------
    # Load ONE MC-Dropout model
    # -------------------------------------------------
    print(
        f"\nLoading MC-Dropout model:\n"
        f"{MC_DROPOUT_MODEL}"
    )

    mc_model = tf.keras.models.load_model(
        MC_DROPOUT_MODEL,
        compile=False
    )

    # -------------------------------------------------
    # Load FIVE ensemble models
    # -------------------------------------------------
    print("\nLoading 5 ensemble models...")

    ensemble_models = {
        name: tf.keras.models.load_model(
            path,
            compile=False
        )
        for name, path in ENSEMBLE_MODELS.items()
    }

    # -------------------------------------------------
    # MC-Dropout: only clean vs adversarial
    # -------------------------------------------------
    mc_results_by_setting = {}

    for data_setting in DATA_SETTINGS:

        mc_results_by_setting[
            data_setting
        ] = evaluate_mc_dropout(
            mc_model,
            x_test,
            y_test,
            data_setting
        )

    del mc_model
    tf.keras.backend.clear_session()

    # -------------------------------------------------
    # Deep Ensemble: clean/adversarial × exclusive/inclusive
    # -------------------------------------------------
    ensemble_results_by_setting = {}

    for data_setting in DATA_SETTINGS:

        for ensemble_setting in ENSEMBLE_SETTINGS:

            setting_key = (
                f"{data_setting}_{ensemble_setting}"
            )

            ensemble_results_by_setting[
                setting_key
            ] = evaluate_deep_ensemble(
                ensemble_models,
                x_test,
                y_test,
                data_setting,
                ensemble_setting
            )

    # -------------------------------------------------
    # Comparison
    # -------------------------------------------------
    comparison = print_comparison(
        mc_results_by_setting,
        ensemble_results_by_setting
    )

    # -------------------------------------------------
    # Save ALL FOUR SETTINGS into ONE combined JSON at the end.
    # No separate JSON file is written for any individual setting.
    # -------------------------------------------------
    output = {
        "dataset": "Fashion-MNIST",

        "protocol": {
            "mc_dropout": {
                "num_models": 1,
                "model": MC_DROPOUT_MODEL,
                "dropout_rate": 0.5,
                "T": T,
            },
            "deep_ensemble": {
                "num_models": 5,
                "seeds": ENSEMBLE_SEEDS,
                "settings": ENSEMBLE_SETTINGS,
            },
            "data_settings": DATA_SETTINGS,
            "comparison_settings": [
                "clean_exclusive",
                "clean_inclusive",
                "adversarial_exclusive",
                "adversarial_inclusive",
            ],
        },

        "mc_dropout_results": mc_results_by_setting,

        "ensemble_results": ensemble_results_by_setting,

        "comparison": comparison,
    }

    output_path = (
        f"{BASE}/mc_dropout_1model_vs_ensemble_5models_"
        f"fashion_four_settings_results_COMBINED.json"
    )

    with open(
        output_path,
        "w"
    ) as f:
        json.dump(
            output,
            f,
            indent=2
        )

    total_time = (
        time.time() - total_start
    )

    print(
        "\nONE COMBINED JSON saved for all four settings:"
    )
    print(output_path)

    print(
        f"\nTotal wall-clock time: "
        f"{total_time:.2f} s "
        f"({total_time / 60:.2f} min)"
    )


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    main()
