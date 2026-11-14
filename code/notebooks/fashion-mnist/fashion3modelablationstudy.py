import os

# Thread caps BEFORE NumPy import
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import time
import json
import random
import numpy as np
import tensorflow as tf


# =====================================================
# CONFIG
# =====================================================

BATCH = 256
SEED = 42

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Models to include in the experiment
# Change ONLY this list for a different set of trained models.
# Example:
#MODEL_SEEDS = [0, 1, 2, 3, 4]
MODEL_SEEDS = [0, 1, 2]
#MODEL_SEEDS = [0, 2, 5, 7, 9]
#MODEL_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if len(MODEL_SEEDS) < 2:
    raise ValueError(
        "At least 2 models are required."
    )

if len(set(MODEL_SEEDS)) != len(MODEL_SEEDS):
    raise ValueError(
        "MODEL_SEEDS contains duplicate seeds."
    )

MODELS = {
    f"seed{s}": f"{BASE}/models/fashion_seed{s}.h5"
    for s in MODEL_SEEDS
}

ADV_DIR = f"{BASE}/adv_data_fashion"

# DeepGini Table 5 parameter grid
NAC_TS = [0.0, 0.75]
NBC_STDS = [0.0, 0.5, 1.0]
SNAC_STDS = [0.0, 0.5, 1.0]
TKNC_KS = [1, 2, 3]
KMNC_BINS_LIST = [1000, 10000]

# None = greedy until no additional coverage gain
CAM_MAX_PICKS = None

# Ensemble setting:
# "exclusive" = exclude the current oracle from the ensemble
# "inclusive" = include the current oracle in the ensemble
ENSEMBLE_SETTING = "inclusive"

if ENSEMBLE_SETTING not in ("exclusive", "inclusive"):
    raise ValueError(
        "ENSEMBLE_SETTING must be 'exclusive' or 'inclusive'."
    )

# Data setting is assigned inside main():
# "clean"       = clean test inputs only
# "adversarial" = clean + model-specific FGSM + BIM
DATA_SETTING = "clean"

if DATA_SETTING not in ("clean", "adversarial"):
    raise ValueError(
        "DATA_SETTING must be 'clean' or 'adversarial'."
    )

rng = np.random.default_rng(SEED)


# =====================================================
# VERIFY MODELS
# =====================================================

def verify_models(model_paths):
    """Check that all configured model files exist and are unique."""
    print("\nChecking model files...")

    missing = []
    real_paths = []

    for name, path in model_paths.items():
        print(f"{name}: {path}")

        if not os.path.isfile(path):
            missing.append(path)

        real_paths.append(os.path.realpath(path))

    if missing:
        raise FileNotFoundError(
            "\nMissing model file(s):\n" +
            "\n".join(missing)
        )

    if len(set(real_paths)) != len(real_paths):
        raise RuntimeError(
            "Duplicate model paths detected. "
            "Each configured model must be independent."
        )

    print(f"\nVerified {len(model_paths)} unique model files.")


# =====================================================
# DATA
# =====================================================

def load_fashion():
    (xtr, ytr), (xte, yte) = tf.keras.datasets.fashion_mnist.load_data()

    xtr = np.expand_dims(
        xtr.astype("float32") / 255.0,
        -1
    )

    xte = np.expand_dims(
        xte.astype("float32") / 255.0,
        -1
    )

    return (
        (xtr, ytr.astype("int64")),
        (xte, yte.astype("int64"))
    )


def load_adv_for(seed):
    """
    Load FGSM and BIM adversarial samples for the current oracle.
    They are later mixed with the clean test set.
    """
    xs = []

    for attack in ["fgsm", "bim"]:
        path = f"{ADV_DIR}/fashion_{attack}_{seed}.npy"

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing adversarial file: {path}"
            )

        xs.append(
            np.load(path).astype("float32")
        )

    return np.concatenate(xs, axis=0)



# =====================================================
# ACTIVATION EXTRACTION
# =====================================================

def build_activation_model(model):
    """
    Extract post-activation outputs.
    Conv2D is followed by a separate Activation layer,
    while Dense layers contain their activation internally.
    """
    layers = model.layers
    outs = []

    for i, layer in enumerate(layers):

        if isinstance(
            layer,
            (tf.keras.layers.Conv2D, tf.keras.layers.Dense)
        ):
            nxt = layers[i + 1] if i + 1 < len(layers) else None

            if isinstance(nxt, tf.keras.layers.Activation):
                outs.append(nxt.output)
            else:
                outs.append(layer.output)

    # Remove final softmax output
    if len(outs) > 1:
        outs = outs[:-1]

    return tf.keras.models.Model(
        inputs=model.inputs,
        outputs=outs
    )


def extract_activations(
    act_model,
    x,
    batch=BATCH
):
    outs = act_model.predict(
        x,
        batch_size=batch,
        verbose=0
    )

    if not isinstance(outs, list):
        outs = [outs]

    feats = []
    layer_ids = []

    for i, o in enumerate(outs):

        if o.ndim == 4:
            o = o.mean(axis=(1, 2))

        feats.append(
            o.astype("float32")
        )

        layer_ids.append(
            np.full(
                o.shape[1],
                i,
                dtype=np.int32
            )
        )

    return (
        np.concatenate(feats, axis=1),
        np.concatenate(layer_ids)
    )


# =====================================================
# RANKING HELPERS
# =====================================================

def rank_desc(scores, rng):
    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    return np.lexsort(
        (
            rng.random(len(scores)),
            -scores
        )
    )


def rank_asc(scores, rng):
    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    return np.lexsort(
        (
            rng.random(len(scores)),
            scores
        )
    )


# =====================================================
# COVERAGE HELPERS
# =====================================================

def greedy_cam(
    cover_bool,
    rng,
    max_picks=None
):
    """
    Vectorized greedy set-cover.
    Returns:
        ranking,
        number of greedy picks,
        coverage rate
    """

    N, M = cover_bool.shape

    covered = np.zeros(
        M,
        dtype=bool
    )

    remaining = np.ones(
        N,
        dtype=bool
    )

    order = []

    while True:

        if (
            max_picks is not None
            and len(order) >= max_picks
        ):
            break

        gains = np.count_nonzero(
            cover_bool & ~covered,
            axis=1
        )

        gains[~remaining] = -1

        best = int(
            gains.argmax()
        )

        if gains[best] <= 0:
            break

        order.append(best)

        covered |= cover_bool[best]

        remaining[best] = False

    rest = np.where(
        remaining
    )[0]

    rng.shuffle(rest)

    full = np.concatenate(
        [
            np.array(order, dtype=int),
            rest
        ]
    )

    return (
        full,
        len(order),
        covered.mean()
    )


def greedy_cam_kmnc(
    bin_ids,
    k_bins,
    rng,
    max_picks=None
):
    """
    Memory-safe KMNC-CAM.
    """

    N, M = bin_ids.shape

    covered = np.zeros(
        (M, k_bins + 2),
        dtype=bool
    )

    remaining = np.ones(
        N,
        dtype=bool
    )

    neuron_idx = np.arange(M)

    order = []

    while True:

        if (
            max_picks is not None
            and len(order) >= max_picks
        ):
            break

        gains = np.count_nonzero(
            ~covered[
                neuron_idx[None, :],
                bin_ids
            ],
            axis=1
        )

        gains[~remaining] = -1

        best = int(
            gains.argmax()
        )

        if gains[best] <= 0:
            break

        order.append(best)

        covered[
            neuron_idx,
            bin_ids[best]
        ] = True

        remaining[best] = False

    rest = np.where(
        remaining
    )[0]

    rng.shuffle(rest)

    full = np.concatenate(
        [
            np.array(order, dtype=int),
            rest
        ]
    )

    cov_rate = (
        covered[:, 1:k_bins + 1].sum()
        / float(M * k_bins)
    )

    return (
        full,
        len(order),
        cov_rate
    )


# =====================================================
# COVERAGE CRITERIA
# =====================================================

def nac_cover(A, t):
    return (
        1.0 / (1.0 + np.exp(-A))
    ) > t


def snac_cover(A, upper):
    return A > upper


def nbc_cover(
    A,
    upper,
    lower
):
    return np.concatenate(
        [
            A > upper,
            A < lower
        ],
        axis=1
    )


def tknc_cover(
    A,
    layer_ids,
    k
):
    N, M = A.shape

    cov = np.zeros(
        (N, M),
        dtype=bool
    )

    for lid in np.unique(layer_ids):

        cols = np.where(
            layer_ids == lid
        )[0]

        sub = A[:, cols]

        kk = min(
            k,
            sub.shape[1]
        )

        top = np.argpartition(
            -sub,
            kk - 1,
            axis=1
        )[:, :kk]

        rows = np.repeat(
            np.arange(N),
            kk
        )

        cov[
            rows,
            cols[top.ravel()]
        ] = True

    return cov


def kmnc_bin_ids(
    A,
    lower,
    upper,
    k_bins
):
    span = np.maximum(
        upper - lower,
        1e-12
    )

    ids = np.floor(
        (A - lower)
        / span
        * k_bins
    ).astype(np.int32) + 1

    return np.clip(
        ids,
        0,
        k_bins + 1
    )


# =====================================================
# UNCERTAINTY METRICS
# =====================================================

def deepgini(p):
    return (
        1 - np.sum(
            p ** 2,
            axis=1
        )
    )


def entropy(p):
    return -np.sum(
        p * np.log(
            p + 1e-12
        ),
        axis=1
    )


def ensemble_variance(probs):
    return np.mean(
        np.var(
            probs,
            axis=0
        ),
        axis=1
    )


def disagreement(mean_probs):
    return (
        1 - np.max(
            mean_probs,
            axis=1
        )
    )


def mutual_information(ens):
    mean_probs = np.mean(
        ens,
        axis=0
    )

    entropy_mean = -np.sum(
        mean_probs
        * np.log(mean_probs + 1e-12),
        axis=1
    )

    entropy_models = -np.sum(
        ens
        * np.log(ens + 1e-12),
        axis=2
    )

    return (
        entropy_mean
        - np.mean(
            entropy_models,
            axis=0
        )
    )


def softmax_confidence(p):
    return np.max(
        p,
        axis=1
    )


def pcs(p):
    sorted_p = -np.sort(
        -p,
        axis=1
    )

    return (
        sorted_p[:, 0]
        - sorted_p[:, 1]
    )


def normalize(x):
    return (
        (x - x.min())
        / (
            x.max()
            - x.min()
            + 1e-12
        )
    )


# =====================================================
# APFD
# =====================================================

def APFD(
    rank,
    faulty
):
    n = len(faulty)

    m = int(
        faulty.sum()
    )

    if m == 0:
        return 0.0

    positions = np.empty(
        n,
        dtype=np.int64
    )

    positions[
        np.asarray(rank)
    ] = np.arange(
        1,
        n + 1
    )

    return (
        1
        - positions[
            faulty.astype(bool)
        ].sum()
        / (n * m)
        + 1 / (2 * n)
    )


# =====================================================
# MAIN
# =====================================================

def run_experiment():

    program_start = time.time()

    print(
        "\n"
        "==============================================================\n"
        f"ROTATING ORACLE | {len(MODEL_SEEDS)}-MODEL ENSEMBLE | {ENSEMBLE_SETTING.upper()} SETTING\n"
        "==============================================================\n"
    )

    # -------------------------------------------------
    # Verify models
    # -------------------------------------------------

    verify_models(MODELS)

    print(
        f"\nTotal trained models: {len(MODELS)}"
    )

    print(
        f"Exclusive ensemble size per round: "
        f"{len(MODELS) - 1}"
    )

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------

    (x_train, y_train), (
        x_clean,
        y_clean
    ) = load_fashion()

    N_clean = len(x_clean)

    # -------------------------------------------------
    # Load all configured models
    # -------------------------------------------------

    models = {
        name: tf.keras.models.load_model(
            path,
            compile=False
        )
        for name, path in MODELS.items()
    }

    names = list(
        models.keys()
    )

    # -------------------------------------------------
    # Metric names
    # -------------------------------------------------

    unc_names = [
        "DeepGini",
        "Entropy",
        "Softmax",
        "PCS",
        "EnsGini",
        "Variance",
        "Disagreement",
        "MutualInfo",
        "AdaptiveVar",
        "AdaptiveMI",
    ]

    cov_names = []

    for t in NAC_TS:
        cov_names += [
            f"NAC({t})-CTM",
            f"NAC({t})-CAM"
        ]

    for s in NBC_STDS:
        cov_names += [
            f"NBC({s})-CTM",
            f"NBC({s})-CAM"
        ]

    for s in SNAC_STDS:
        cov_names += [
            f"SNAC({s})-CTM",
            f"SNAC({s})-CAM"
        ]

    for k in TKNC_KS:
        cov_names.append(
            f"TKNC({k})-CAM"
        )

    for b in KMNC_BINS_LIST:
        cov_names.append(
            f"KMNC({b})-CAM"
        )

    metric_names = (
        unc_names
        + cov_names
        + ["Random"]
    )

    # -------------------------------------------------
    # Result storage
    # -------------------------------------------------

    results = {
        k: []
        for k in metric_names
    }

    results_by_oracle = {
        oracle: {}
        for oracle in names
    }

    times = {
        k: []
        for k in metric_names
    }

    ranking_times = {
        k: []
        for k in metric_names
    }

    end_to_end_times = {
        k: []
        for k in metric_names
    }

    maxcov = {
        k: []
        for k in cov_names
    }

    picks = {
        k: []
        for k in cov_names
        if k.endswith("CAM")
    }

    # =================================================
    # ROTATING ORACLE LOOP
    # =================================================

    for oracle in names:

        print(
            f"\n{'=' * 62}\n"
            f"ORACLE = {oracle}\n"
            f"{'=' * 62}"
        )

        # -------------------------------------------------
        # BUILD EVALUATION POOL
        # -------------------------------------------------

        if DATA_SETTING == "clean":

            # Clean-only setting: 10,000 clean test inputs.
            x_all = x_clean
            y_all = y_clean
            N = len(x_all)

            print(
                f"Clean samples: {len(x_clean)} | "
                f"Adversarial samples: 0 | "
                f"Total samples: {N}"
            )

        elif DATA_SETTING == "adversarial":

            # Adversarial setting:
            # clean + model-specific FGSM + model-specific BIM.
            x_adv = load_adv_for(oracle)

            if len(x_adv) % len(x_clean) != 0:
                raise ValueError(
                    f"Adversarial data size ({len(x_adv)}) is not "
                    f"an integer multiple of clean test size ({len(x_clean)})."
                )

            num_attacks = len(x_adv) // len(x_clean)

            x_all = np.concatenate(
                [x_clean, x_adv],
                axis=0
            )

            y_all = np.tile(
                y_clean,
                1 + num_attacks
            )

            N = len(x_all)

            print(
                f"Clean samples: {len(x_clean)} | "
                f"Adversarial samples: {len(x_adv)} | "
                f"Total samples: {N}"
            )

        else:
            raise ValueError(
                "DATA_SETTING must be 'clean' or 'adversarial'."
            )

        # -------------------------------------------------
        # Predictions from ALL configured models
        # -------------------------------------------------

        # Time each model inference separately so that end-to-end
        # prioritization time can include only the model predictions
        # required by each prioritization method.
        model_inference_times = {}
        all_probs_list = []

        for n in names:

            t_pred = time.time()

            p = models[n].predict(
                x_all,
                batch_size=BATCH,
                verbose=0
            )

            model_inference_times[n] = time.time() - t_pred
            all_probs_list.append(p)

        all_probs = np.stack(
            all_probs_list,
            axis=0
        )

        # Shape:
        # (K, N, 10)

        print(
            f"all_probs shape = {all_probs.shape}"
        )

        # -------------------------------------------------
        # Select oracle
        # -------------------------------------------------

        oracle_idx = names.index(
            oracle
        )

        oracle_probs = all_probs[
            oracle_idx
        ]

        # -------------------------------------------------
        # Oracle fault labels
        # -------------------------------------------------

        pred = np.argmax(
            oracle_probs,
            axis=1
        )

        faulty = (
            pred != y_all
        ).astype(int)

        print(
            f"Pool: {N} | "
            f"Faulty: {faulty.sum()}"
        )

        # -------------------------------------------------
        # ENSEMBLE
        # Exclusive: exclude current oracle
        # Inclusive: include current oracle
        # -------------------------------------------------

        if ENSEMBLE_SETTING == "exclusive":

            ensemble_mask = (
                np.arange(
                    len(all_probs)
                )
                != oracle_idx
            )

            ens = all_probs[
                ensemble_mask
            ]

        else:
            ens = all_probs

        print(
            f"Oracle: {oracle} | "
            f"Ensemble setting: {ENSEMBLE_SETTING} | "
            f"Ensemble shape: {ens.shape}"
        )

        if ENSEMBLE_SETTING == "exclusive":
            expected_ensemble_size = len(MODEL_SEEDS) - 1
        else:
            expected_ensemble_size = len(MODEL_SEEDS)

        assert ens.shape[0] == expected_ensemble_size, (
            f"Expected {expected_ensemble_size} ensemble models, "
            f"got {ens.shape[0]}"
        )

        mean_probs = np.mean(
            ens,
            axis=0
        )

        oracle_inference_time = model_inference_times[oracle]

        ensemble_names = [
            names[j]
            for j in range(len(names))
            if ENSEMBLE_SETTING == "inclusive" or j != oracle_idx
        ]

        ensemble_inference_time = sum(
            model_inference_times[n]
            for n in ensemble_names
        )

        all_models_inference_time = sum(
            model_inference_times.values()
        )

        rankings = {}

        # =================================================
        # SINGLE-MODEL UNCERTAINTY
        # =================================================

        # DeepGini
        t0 = time.time()

        gini = deepgini(
            oracle_probs
        )

        rank_start = time.time()

        rankings["DeepGini"] = rank_desc(
            gini,
            rng
        )

        ranking_times["DeepGini"].append(
            time.time() - rank_start
        )

        times["DeepGini"].append(
            time.time() - t0
        )

        end_to_end_times["DeepGini"].append(
            oracle_inference_time + times["DeepGini"][-1]
        )

        # Entropy
        t0 = time.time()

        ent = entropy(
            oracle_probs
        )

        rank_start = time.time()

        rankings["Entropy"] = rank_desc(
            ent,
            rng
        )

        ranking_times["Entropy"].append(
            time.time() - rank_start
        )

        times["Entropy"].append(
            time.time() - t0
        )

        end_to_end_times["Entropy"].append(
            oracle_inference_time + times["Entropy"][-1]
        )

        # Softmax
        t0 = time.time()

        soft = softmax_confidence(
            oracle_probs
        )

        rank_start = time.time()

        rankings["Softmax"] = rank_asc(
            soft,
            rng
        )

        ranking_times["Softmax"].append(
            time.time() - rank_start
        )

        times["Softmax"].append(
            time.time() - t0
        )

        end_to_end_times["Softmax"].append(
            oracle_inference_time + times["Softmax"][-1]
        )

        # PCS
        t0 = time.time()

        pcs_s = pcs(
            oracle_probs
        )

        rank_start = time.time()

        rankings["PCS"] = rank_asc(
            pcs_s,
            rng
        )

        ranking_times["PCS"].append(
            time.time() - rank_start
        )

        times["PCS"].append(
            time.time() - t0
        )

        end_to_end_times["PCS"].append(
            oracle_inference_time + times["PCS"][-1]
        )

        # =================================================
        # ENSEMBLE UNCERTAINTY
        # =================================================

        # Ensemble Gini
        t0 = time.time()

        gini_ens = deepgini(
            mean_probs
        )

        rank_start = time.time()

        rankings["EnsGini"] = rank_desc(
            gini_ens,
            rng
        )

        ranking_times["EnsGini"].append(
            time.time() - rank_start
        )

        times["EnsGini"].append(
            time.time() - t0
        )

        end_to_end_times["EnsGini"].append(
            ensemble_inference_time + times["EnsGini"][-1]
        )

        # Variance
        t0 = time.time()

        var = ensemble_variance(
            ens
        )

        rank_start = time.time()

        rankings["Variance"] = rank_desc(
            var,
            rng
        )

        ranking_times["Variance"].append(
            time.time() - rank_start
        )

        times["Variance"].append(
            time.time() - t0
        )

        end_to_end_times["Variance"].append(
            ensemble_inference_time + times["Variance"][-1]
        )

        # Disagreement
        t0 = time.time()

        dis = disagreement(
            mean_probs
        )

        rank_start = time.time()

        rankings["Disagreement"] = rank_desc(
            dis,
            rng
        )

        ranking_times["Disagreement"].append(
            time.time() - rank_start
        )

        times["Disagreement"].append(
            time.time() - t0
        )

        end_to_end_times["Disagreement"].append(
            ensemble_inference_time + times["Disagreement"][-1]
        )

        # Mutual Information
        t0 = time.time()

        mi = mutual_information(
            ens
        )

        rank_start = time.time()

        rankings["MutualInfo"] = rank_desc(
            mi,
            rng
        )

        ranking_times["MutualInfo"].append(
            time.time() - rank_start
        )

        times["MutualInfo"].append(
            time.time() - t0
        )

        end_to_end_times["MutualInfo"].append(
            ensemble_inference_time + times["MutualInfo"][-1]
        )

        # Adaptive Variance
        t0 = time.time()

        adaptive_var = (
            (1 - normalize(var))
            * normalize(gini)
            + normalize(var) ** 2
        )

        rank_start = time.time()

        rankings["AdaptiveVar"] = rank_desc(
            adaptive_var,
            rng
        )

        ranking_times["AdaptiveVar"].append(
            time.time() - rank_start
        )

        times["AdaptiveVar"].append(
            time.time() - t0
        )

        end_to_end_times["AdaptiveVar"].append(
            all_models_inference_time + times["AdaptiveVar"][-1]
        )

        # Adaptive MI
        t0 = time.time()

        adaptive_mi = (
            0.5
            * normalize(
                entropy(oracle_probs)
            )
            +
            0.5
            * normalize(
                mutual_information(ens)
            )
        )

        rank_start = time.time()

        rankings["AdaptiveMI"] = rank_desc(
            adaptive_mi,
            rng
        )

        ranking_times["AdaptiveMI"].append(
            time.time() - rank_start
        )

        times["AdaptiveMI"].append(
            time.time() - t0
        )

        end_to_end_times["AdaptiveMI"].append(
            all_models_inference_time + times["AdaptiveMI"][-1]
        )

        # =================================================
        # ACTIVATIONS
        # =================================================

        t_ex0 = time.time()

        act_model = build_activation_model(
            models[oracle]
        )

        A_train, _ = extract_activations(
            act_model,
            x_train
        )

        std_train = A_train.std(
            axis=0
        )

        max_train = A_train.max(
            axis=0
        )

        min_train = A_train.min(
            axis=0
        )

        A_all, layer_ids = extract_activations(
            act_model,
            x_all
        )

        activation_prep_time = (
            time.time()
            - t_ex0
        )

        print(
            f"Neurons: {A_all.shape[1]} | "
            f"extraction {activation_prep_time:.1f}s"
        )

        # =================================================
        # CTM / CAM HELPER
        # =================================================

        def do_ctm_cam(tag, cov, cov_compute_time=0.0):

            # CTM
            t0 = time.time()

            rank_start = time.time()

            rankings[
                f"{tag}-CTM"
            ] = rank_desc(
                cov.sum(axis=1),
                rng
            )

            ranking_times[
                f"{tag}-CTM"
            ].append(
                time.time() - rank_start
            )

            times[
                f"{tag}-CTM"
            ].append(
                time.time()
                - t0
            )

            end_to_end_times[
                f"{tag}-CTM"
            ].append(
                activation_prep_time
                + cov_compute_time
                + times[f"{tag}-CTM"][-1]
            )

            # CAM
            t0 = time.time()

            rank_start = time.time()

            r, n_p, cr = greedy_cam(
                cov,
                rng,
                CAM_MAX_PICKS
            )

            ranking_times[
                f"{tag}-CAM"
            ].append(
                time.time() - rank_start
            )

            rankings[
                f"{tag}-CAM"
            ] = r

            times[
                f"{tag}-CAM"
            ].append(
                
                 time.time()
                - t0
            )

            end_to_end_times[
                f"{tag}-CAM"
            ].append(
                activation_prep_time
                + cov_compute_time
                + times[f"{tag}-CAM"][-1]
            )

            maxcov[
                f"{tag}-CTM"
            ].append(cr)

            maxcov[
                f"{tag}-CAM"
            ].append(cr)

            picks[
                f"{tag}-CAM"
            ].append(n_p)

        # =================================================
        # NAC
        # =================================================

        for t in NAC_TS:

            t_cov0 = time.time()

            cov = nac_cover(
                A_all,
                t
            )

            cov_compute_time = time.time() - t_cov0

            do_ctm_cam(
                f"NAC({t})",
                cov,
                cov_compute_time
            )

        # =================================================
        # NBC
        # =================================================

        for s in NBC_STDS:

            t_cov0 = time.time()

            cov = nbc_cover(
                A_all,
                max_train
                + s * std_train,
                min_train
                - s * std_train
            )

            cov_compute_time = time.time() - t_cov0

            do_ctm_cam(
                f"NBC({s})",
                cov,
                cov_compute_time
            )

        # =================================================
        # SNAC
        # =================================================

        for s in SNAC_STDS:

            t_cov0 = time.time()

            cov = snac_cover(
                A_all,
                max_train
                + s * std_train
            )

            cov_compute_time = time.time() - t_cov0

            do_ctm_cam(
                f"SNAC({s})",
                cov,
                cov_compute_time
            )

        # =================================================
        # TKNC
        # =================================================

        for k in TKNC_KS:

            t_cov0 = time.time()

            cov = tknc_cover(
                A_all,
                layer_ids,
                k
            )

            cov_compute_time = time.time() - t_cov0

            t0 = time.time()

            rank_start = time.time()

            r, n_p, cr = greedy_cam(
                cov,
                rng,
                CAM_MAX_PICKS
            )

            ranking_times[
                f"TKNC({k})-CAM"
            ].append(
                time.time() - rank_start
            )

            rankings[
                f"TKNC({k})-CAM"
            ] = r

            times[
                f"TKNC({k})-CAM"
            ].append(
            
                time.time()
                - t0
            )

            end_to_end_times[
                f"TKNC({k})-CAM"
            ].append(
                activation_prep_time
                + cov_compute_time
                + times[f"TKNC({k})-CAM"][-1]
            )

            maxcov[
                f"TKNC({k})-CAM"
            ].append(cr)

            picks[
                f"TKNC({k})-CAM"
            ].append(n_p)

        # =================================================
        # KMNC
        # =================================================

        for b in KMNC_BINS_LIST:

            t_cov0 = time.time()

            bins = kmnc_bin_ids(
                A_all,
                min_train,
                max_train,
                b
            )

            cov_compute_time = time.time() - t_cov0

            t0 = time.time()

            rank_start = time.time()

            r, n_p, cr = greedy_cam_kmnc(
                bins,
                b,
                rng,
                CAM_MAX_PICKS
            )

            ranking_times[
                f"KMNC({b})-CAM"
            ].append(
                time.time() - rank_start
            )

            rankings[
                f"KMNC({b})-CAM"
            ] = r

            elapsed = (
            
                time.time()
                - t0
            )

            times[
                f"KMNC({b})-CAM"
            ].append(
                elapsed
            )

            end_to_end_times[
                f"KMNC({b})-CAM"
            ].append(
                activation_prep_time
                + cov_compute_time
                + times[f"KMNC({b})-CAM"][-1]
            )

            maxcov[
                f"KMNC({b})-CAM"
            ].append(cr)

            picks[
                f"KMNC({b})-CAM"
            ].append(n_p)

        # =================================================
        # RANDOM
        # =================================================

        t0 = time.time()

        rank_start = time.time()

        rankings["Random"] = (
            rng.permutation(N)
        )

        ranking_times["Random"].append(
            time.time() - rank_start
        )

        times["Random"].append(
            time.time() - t0
        )

        end_to_end_times["Random"].append(
            times["Random"][-1]
        )

        # =================================================
        # APFD FOR CURRENT ORACLE
        # =================================================

        for metric in metric_names:

            apfd_value = APFD(
                rankings[metric],
                faulty
            )

            results[metric].append(
                apfd_value
            )

            results_by_oracle[
                oracle
            ][metric] = apfd_value

        # =================================================
        # PRINT CURRENT ORACLE
        # =================================================

        print(
            "\n------ APFD FOR CURRENT ORACLE ------"
        )

        for metric in metric_names:

            print(
                f"{metric:18s}: "
                f"{results_by_oracle[oracle][metric]:.4f}"
            )

    # =====================================================
    # SAVE PER-ORACLE RESULTS
    # =====================================================

    OUTPUT_JSON = (
        f"apfd_by_oracle_{len(MODEL_SEEDS)}models_"
        f"{DATA_SETTING}_{ENSEMBLE_SETTING}.json"
    )

    with open(
        OUTPUT_JSON,
        "w"
    ) as f:

        json.dump(
            results_by_oracle,
            f,
            indent=2
        )

    print(
        f"\nSaved per-oracle APFD results to: "
        f"{OUTPUT_JSON}"
    )

    # -------------------------------------------------
    # SAVE PER-ORACLE SCORE TIME
    # -------------------------------------------------
    # Keep the original APFD JSON unchanged.
    # Save score time separately so APFD files remain
    # backward-compatible with the existing analysis code.

    time_by_oracle = {
        oracle: {}
        for oracle in names
    }

    for oracle in names:
        oracle_idx_tmp = names.index(oracle)

        # The time lists are appended in oracle order.
        # Each metric therefore has one time value per oracle.
        oracle_pos = oracle_idx_tmp

        for metric in metric_names:
            time_by_oracle[oracle][metric] = times[metric][oracle_pos]

    TIME_JSON = (
        f"time_by_oracle_{len(MODEL_SEEDS)}models_"
        f"{DATA_SETTING}_{ENSEMBLE_SETTING}.json"
    )

    with open(
        TIME_JSON,
        "w"
    ) as f:

        json.dump(
            time_by_oracle,
            f,
            indent=2
        )

    print(
        f"Saved per-oracle SCORE TIME results to: "
        f"{TIME_JSON}"
    )

    # -------------------------------------------------
    # SAVE PER-ORACLE END-TO-END TIME
    # -------------------------------------------------

    end_to_end_time_by_oracle = {
        oracle: {}
        for oracle in names
    }

    for oracle in names:
        oracle_pos = names.index(oracle)

        for metric in metric_names:
            end_to_end_time_by_oracle[oracle][metric] = (
                end_to_end_times[metric][oracle_pos]
            )

    END_TO_END_TIME_JSON = (
        f"end_to_end_time_by_oracle_{len(MODEL_SEEDS)}models_"
        f"{DATA_SETTING}_{ENSEMBLE_SETTING}.json"
    )

    with open(
        END_TO_END_TIME_JSON,
        "w"
    ) as f:

        json.dump(
            end_to_end_time_by_oracle,
            f,
            indent=2
        )

    print(
        f"Saved per-oracle END-TO-END TIME results to: "
        f"{END_TO_END_TIME_JSON}"
    )

    # -------------------------------------------------
    # SAVE COMBINED APFD + END-TO-END TIME
    # -------------------------------------------------
    # This file is convenient for generating the final
    # APFD + Time LaTeX tables without changing the
    # original APFD-only JSON format.

    combined_results = {
        oracle: {}
        for oracle in names
    }

    for oracle in names:
        oracle_pos = names.index(oracle)

        for metric in metric_names:
            combined_results[oracle][metric] = {
                "APFD": results_by_oracle[oracle][metric],
                "Time": end_to_end_times[metric][oracle_pos]
            }

    COMBINED_JSON = (
        f"apfd_time_by_oracle_{len(MODEL_SEEDS)}models_"
        f"{DATA_SETTING}_{ENSEMBLE_SETTING}.json"
    )

    with open(
        COMBINED_JSON,
        "w"
    ) as f:

        json.dump(
            combined_results,
            f,
            indent=2
        )

    print(
        f"Saved combined APFD + END-TO-END TIME results to: "
        f"{COMBINED_JSON}"
    )

    # =====================================================
    # FINAL APFD
    # =====================================================

    print(
        "\n"
        + "=" * 62
    )

    print(
        f"====== APFD | {len(MODEL_SEEDS)} MODELS | "
        f"{DATA_SETTING.upper()} | {ENSEMBLE_SETTING.upper()} ======\n"
    )

    for metric in metric_names:

        mean_apfd = np.mean(
            results[metric]
        )

        std_apfd = np.std(
            results[metric]
        )

        line = (
            f"{metric:18s}: "
            f"{mean_apfd:.4f} ± "
            f"{std_apfd:.4f}"
        )

        if metric in maxcov:

            line += (
                f"   maxcov="
                f"{np.mean(maxcov[metric]) * 100:5.1f}%"
            )

        if metric in picks:

            line += (
                f"/{np.mean(picks[metric]):.0f}"
            )

        print(line)

    # =====================================================
    # FINAL TIME
    # =====================================================

    print(
        "\n"
        + "=" * 62
    )

    print(
        f"====== SCORE TIME (s) | {len(MODEL_SEEDS)} MODELS | "
        f"{DATA_SETTING.upper()} | {ENSEMBLE_SETTING.upper()} ======\n"
    )

    for metric in metric_names:

        print(
            f"{metric:18s}: "
            f"{np.mean(times[metric]):.4f} ± "
            f"{np.std(times[metric]):.4f}"
        )

    # =====================================================
    # END-TO-END PRIORITIZATION TIME
    # =====================================================

    print(
        "\n"
        + "=" * 62
    )

    print(
        f"====== END-TO-END PRIORITIZATION TIME (s) | "
        f"{len(MODEL_SEEDS)} MODELS | "
        f"{DATA_SETTING.upper()} | {ENSEMBLE_SETTING.upper()} ======\n"
    )

    for metric in metric_names:

        print(
            f"{metric:18s}: "
            f"{np.mean(end_to_end_times[metric]):.4f} ± "
            f"{np.std(end_to_end_times[metric]):.4f}"
        )


    # =====================================================
    # RANKING TIME
    # =====================================================

    print(
        "\n"
        + "=" * 62
    )

    print(
        f"====== RANKING TIME (s) | "
        f"{len(MODEL_SEEDS)} MODELS | "
        f"{DATA_SETTING.upper()} | {ENSEMBLE_SETTING.upper()} ======\n"
    )

    for metric in metric_names:
        print(
            f"{metric:18s}: "
            f"{np.mean(ranking_times[metric]):.6f} ± "
            f"{np.std(ranking_times[metric]):.6f}"
        )

    # =====================================================
    # COMPLETE EXPERIMENT WALL-CLOCK TIME
    # =====================================================

    total_time = time.time() - program_start

    print("\n" + "=" * 62)
    print(
        f"====== TOTAL WALL-CLOCK TIME | "
        f"{len(MODEL_SEEDS)} MODELS | "
        f"{DATA_SETTING.upper()} | {ENSEMBLE_SETTING.upper()} ======"
    )
    print(
        f"{total_time:.2f} seconds "
        f"({total_time / 60:.2f} minutes)"
    )
    print("=" * 62)


def main():

    # =====================================================
    # MODEL-SIZE STUDY × FOUR SETTINGS
    # =====================================================
    #
    # Model sizes:
    #   3 models   (ablation: below-5 degradation)
    #   5 models
    #   7 models
    #   10 models
    #
    # For EACH model size, run:
    #   1. Clean + Exclusive
    #   2. Clean + Inclusive
    #   3. Adversarial + Exclusive
    #   4. Adversarial + Inclusive
    #
    # Note: with 3 models the exclusive setting leaves 2 estimators,
    # which is the minimum for MI / variance (a 2-model ensemble would
    # leave a single estimator, for which those metrics are undefined).
    # =====================================================

    experiment_model_sets = [
        [0, 1, 2],                           # 3 models
        [0, 1, 2, 3, 4],                     # 5 models
        [0, 1, 2, 3, 4, 5, 6],               # 7 models
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],      # 10 models
    ]

    data_settings = [
        "clean",
        "adversarial",
    ]

    ensemble_settings = [
        "exclusive",
        "inclusive",
    ]

    overall_start = time.time()

    for model_seeds in experiment_model_sets:

        global MODEL_SEEDS, MODELS, rng
        global ENSEMBLE_SETTING, DATA_SETTING

        MODEL_SEEDS = list(model_seeds)

        MODELS = {
            f"seed{s}": f"{BASE}/models/fashion_seed{s}.h5"
            for s in MODEL_SEEDS
        }

        # -------------------------------------------------
        # Run all 4 settings for this model size
        # -------------------------------------------------

        for data_setting in data_settings:

            for ensemble_setting in ensemble_settings:

                DATA_SETTING = data_setting
                ENSEMBLE_SETTING = ensemble_setting

                # Reset RNG for reproducibility of every run.
                rng = np.random.default_rng(SEED)

                print("\n\n" + "#" * 78)
                print(
                    f"STARTING EXPERIMENT: "
                    f"{len(MODEL_SEEDS)} MODELS | "
                    f"{DATA_SETTING.upper()} DATA | "
                    f"{ENSEMBLE_SETTING.upper()} ENSEMBLE"
                )
                print("#" * 78)

                run_experiment()

                print("#" * 78)
                print(
                    f"FINISHED EXPERIMENT: "
                    f"{len(MODEL_SEEDS)} MODELS | "
                    f"{DATA_SETTING.upper()} DATA | "
                    f"{ENSEMBLE_SETTING.upper()} ENSEMBLE"
                )
                print("#" * 78)

    overall_time = time.time() - overall_start

    print("\n" + "=" * 78)
    print("====== ALL MODEL-SIZE × FOUR-SETTING EXPERIMENTS COMPLETED ======")
    print("=" * 78)
    print(
        f"Total wall-clock time for all 12 runs: "
        f"{overall_time:.2f} seconds "
        f"({overall_time / 60:.2f} minutes)"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()