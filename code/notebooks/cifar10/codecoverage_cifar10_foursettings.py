"""
CIFAR-10 (ResNet-20) rotating-oracle TCP comparison:
    uncertainty / ensemble metrics  vs  coverage-based baselines

Runs all four configurations (clean/adversarial x exclusive/inclusive),
saves per-run JSON with APFD + end-to-end prioritization time + wall-clock, writes a
combined JSON, and auto-generates one LaTeX table with all four settings.

Architecture handling matches the ORIGINAL scripts exactly:
  * normalization  : subtract per-pixel training mean, applied to BOTH the
                     train activations used for bounds AND the evaluation pool.
                     Adversarial files are stored in [0,1] space and get the
                     SAME single mean-subtraction (matches the reference
                     load_adv_for), so clean and adversarial live in one space.
  * models         : resnet20_best / seed1 / seed2 / seed9 / seed23
  * adv mapping    : cifar10_{attack}_resnet20_{name}.npy, name = best/seedX
  
import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

# =====================================================
# CONFIG
# =====================================================

BATCH = 128
SEED = 42
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# model name -> (model file, adv-file suffix)  -- matches original scripts
MODEL_SPEC = {
    "best":   ("resnet20_best.h5",   "resnet20_best"),
    "seed1":  ("resnet20_seed1.h5",  "resnet20_seed1"),
    "seed2":  ("resnet20_seed2.h5",  "resnet20_seed2"),
    "seed9":  ("resnet20_seed9.h5",  "resnet20_seed9"),
    "seed23": ("resnet20_seed23.h5", "resnet20_seed23"),
}

MODELS = {k: f"{BASE}/models/{v[0]}" for k, v in MODEL_SPEC.items()}
ADV_SUFFIX = {k: v[1] for k, v in MODEL_SPEC.items()}

ADV_DIR = f"{BASE}/adv_data"

NAC_TS = [0.0, 0.75]
NBC_STDS = [0.0, 0.5, 1.0]
SNAC_STDS = [0.0, 0.5, 1.0]
TKNC_KS = [1, 2, 3]
KMNC_BINS_LIST = [1000]
CAM_MAX_PICKS = None

rng = np.random.default_rng(SEED)

# =====================================================
# DATA  (mean-subtraction, exactly as in training)
# =====================================================

def load_cifar():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten().astype("int64")
    y_test = y_test.flatten().astype("int64")
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    mean = np.mean(x_train, axis=0)          # per-pixel mean over train
    x_train -= mean
    x_test -= mean
    return (x_train, y_train), (x_test, y_test), mean

def load_adv_for(model_key, mean):
    
    suffix = ADV_SUFFIX[model_key]
    xs = []
    for attack in ["fgsm", "bim"]:
        path = f"{ADV_DIR}/cifar10_{attack}_{suffix}.npy"
        if os.path.exists(path):
            x = np.load(path).astype("float32")
            x = x - mean                 # same single normalization as clean pool
            xs.append(x)
        else:
            raise FileNotFoundError(
                f"Missing adversarial file: {path}"
            )
    if len(xs) != 2:
        raise RuntimeError(
            f"Expected both FGSM and BIM adversarial files for oracle '{model_key}'."
        )
    return np.concatenate(xs, axis=0)

# =====================================================
# ACTIVATION EXTRACTION  (ReLU outputs for ResNet)
# =====================================================

def build_activation_model(model):
    
    relu_layers = [
        l for l in model.layers
        if isinstance(l, tf.keras.layers.ReLU)
        or (
            isinstance(l, tf.keras.layers.Activation)
            and l.get_config().get("activation") == "relu"
        )
    ]
    if not relu_layers:
        # fallback: some models bake relu into Conv/Dense
        relu_layers = [l for l in model.layers
                       if isinstance(l, (tf.keras.layers.Conv2D,
                                         tf.keras.layers.Dense))][:-1]
    return tf.keras.models.Model(inputs=model.inputs,
                                 outputs=[l.output for l in relu_layers])

def extract_activations(act_model, x, batch=BATCH):
    outs = act_model.predict(x, batch_size=batch, verbose=0)
    if not isinstance(outs, list):
        outs = [outs]
    feats, layer_ids = [], []
    for i, o in enumerate(outs):
        if o.ndim == 4:                       # (N,H,W,C) -> (N,C)
            o = o.mean(axis=(1, 2))
        feats.append(o.astype("float32"))
        layer_ids.append(np.full(o.shape[1], i, dtype=np.int32))
    return np.concatenate(feats, axis=1), np.concatenate(layer_ids)

# =====================================================
# RANKING HELPERS
# =====================================================

def rank_desc(scores, rng):
    scores = np.asarray(scores, dtype=np.float64)
    return np.lexsort((rng.random(len(scores)), -scores))

def rank_asc(scores, rng):
    scores = np.asarray(scores, dtype=np.float64)
    return np.lexsort((rng.random(len(scores)), scores))

def greedy_cam(cover_bool, rng, max_picks=None):
    N, M = cover_bool.shape
    covered = np.zeros(M, dtype=bool)
    remaining = np.ones(N, dtype=bool)
    order = []
    while True:
        if max_picks is not None and len(order) >= max_picks:
            break
        gains = np.count_nonzero(cover_bool & ~covered, axis=1)
        gains[~remaining] = -1
        best = int(gains.argmax())
        if gains[best] <= 0:
            break
        order.append(best)
        covered |= cover_bool[best]
        remaining[best] = False
    rest = np.where(remaining)[0]
    rng.shuffle(rest)
    return np.concatenate([np.array(order, dtype=int), rest]), len(order), covered.mean()

def greedy_cam_kmnc(bin_ids, k_bins, rng, max_picks=None):
    N, M = bin_ids.shape
    covered = np.zeros((M, k_bins + 2), dtype=bool)
    remaining = np.ones(N, dtype=bool)
    neuron_idx = np.arange(M)
    order = []
    while True:
        if max_picks is not None and len(order) >= max_picks:
            break
        gains = np.count_nonzero(~covered[neuron_idx[None, :], bin_ids], axis=1)
        gains[~remaining] = -1
        best = int(gains.argmax())
        if gains[best] <= 0:
            break
        order.append(best)
        covered[neuron_idx, bin_ids[best]] = True
        remaining[best] = False
    rest = np.where(remaining)[0]
    rng.shuffle(rest)
    full = np.concatenate([np.array(order, dtype=int), rest])
    cov_rate = covered[:, 1:k_bins + 1].sum() / float(M * k_bins)
    return full, len(order), cov_rate

# =====================================================
# COVERAGE CRITERIA
# =====================================================

def nac_cover(A, t):
    return (1.0 / (1.0 + np.exp(-A))) > t

def snac_cover(A, upper):
    return A > upper

def nbc_cover(A, upper, lower):
    return np.concatenate([A > upper, A < lower], axis=1)

def tknc_cover(A, layer_ids, k):
    N, M = A.shape
    cov = np.zeros((N, M), dtype=bool)
    for lid in np.unique(layer_ids):
        cols = np.where(layer_ids == lid)[0]
        sub = A[:, cols]
        kk = min(k, sub.shape[1])
        top = np.argpartition(-sub, kk - 1, axis=1)[:, :kk]
        rows = np.repeat(np.arange(N), kk)
        cov[rows, cols[top.ravel()]] = True
    return cov

def kmnc_bin_ids(A, lower, upper, k_bins):
    span = np.maximum(upper - lower, 1e-12)
    ids = np.floor((A - lower) / span * k_bins).astype(np.int32) + 1
    return np.clip(ids, 0, k_bins + 1)

# =====================================================
# UNCERTAINTY METRICS
# =====================================================

def deepgini(p): return 1 - np.sum(p ** 2, axis=1)
def entropy(p): return -np.sum(p * np.log(p + 1e-12), axis=1)
def ensemble_variance(probs): return np.mean(np.var(probs, axis=0), axis=1)
def disagreement(mp): return 1 - np.max(mp, axis=1)

def mutual_information(ens):
    mp = np.mean(ens, axis=0)
    em = -np.sum(mp * np.log(mp + 1e-12), axis=1)
    emod = -np.sum(ens * np.log(ens + 1e-12), axis=2)
    return em - np.mean(emod, axis=0)

def softmax_confidence(p): return np.max(p, axis=1)
def pcs(p):
    s = -np.sort(-p, axis=1); return s[:, 0] - s[:, 1]
def normalize(x): return (x - x.min()) / (x.max() - x.min() + 1e-12)

# =====================================================
# APFD
# =====================================================

def APFD(rank, faulty):
    n = len(faulty); m = int(faulty.sum())
    if m == 0: return 0.0
    pos = np.empty(n, dtype=np.int64)
    pos[np.asarray(rank)] = np.arange(1, n + 1)
    return 1 - pos[faulty.astype(bool)].sum() / (n * m) + 1 / (2 * n)

# =====================================================
# EXPERIMENT RUNNER
# =====================================================

RUN_CONFIGS = [
    ("clean", "exclusive"),
    ("clean", "inclusive"),
    ("adversarial", "exclusive"),
    ("adversarial", "inclusive"),
]

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def build_metric_names():
    unc_names = [
        "DeepGini", "Entropy", "Softmax", "PCS",
        "EnsGini", "Variance", "Disagreement", "MutualInfo",
, "AdaptiveMI"
    ]

    cov_names = []
    for t in NAC_TS:
        cov_names += [f"NAC({t})-CTM", f"NAC({t})-CAM"]
    for s in NBC_STDS:
        cov_names += [f"NBC({s})-CTM", f"NBC({s})-CAM"]
    for s in SNAC_STDS:
        cov_names += [f"SNAC({s})-CTM", f"SNAC({s})-CAM"]
    for k in TKNC_KS:
        cov_names.append(f"TKNC({k})-CAM")
    for b in KMNC_BINS_LIST:
        cov_names.append(f"KMNC({b})-CAM")

    return unc_names, cov_names, unc_names + cov_names + ["Random"]

def prepare_pool(dataset_mode, oracle, x_clean, y_clean, mean):
    if dataset_mode == "clean":
        return x_clean, y_clean

    if dataset_mode != "adversarial":
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")

    x_adv = load_adv_for(oracle, mean)
    if x_adv is None:
        raise FileNotFoundError(
            f"No adversarial data found for oracle '{oracle}'. "
            "The adversarial run requires both FGSM/BIM data."
        )

    if len(x_adv) % len(x_clean) != 0:
        raise ValueError(
            f"Adversarial data size ({len(x_adv)}) is not an integer multiple "
            f"of clean test size ({len(x_clean)})."
        )

    num_attacks = len(x_adv) // len(x_clean)

    # Adversarial experiment pool = CLEAN + FGSM + BIM.
    # For CIFAR-10: 10,000 clean + 10,000 FGSM + 10,000 BIM = 30,000.
    x_all = np.concatenate([x_clean, x_adv], axis=0)
    y_all = np.tile(y_clean, 1 + num_attacks)

    return x_all, y_all

def run_one_configuration(
    dataset_mode,
    ensemble_setting,
    models,
    names,
    x_train,
    x_clean,
    y_clean,
    mean,
):
    """Run one of the four experiments and save its JSON results."""

    run_start_wall = time.perf_counter()
    started_at = utc_now()
    rng_local = np.random.default_rng(SEED)

    unc_names, cov_names, metric_names = build_metric_names()

    results = {k: [] for k in metric_names}
    times = {k: [] for k in metric_names}
    end_to_end_times = {k: [] for k in metric_names}
    maxcov = {k: [] for k in cov_names}
    picks = {k: [] for k in cov_names if k.endswith("CAM")}

    per_oracle = {}

    print("\n" + "#" * 78)
    print(
        f"RUN: {dataset_mode.upper()} | {ensemble_setting.upper()} | "
        f"{len(names)}-MODEL ENSEMBLE"
    )
    print("#" * 78)

    for oracle in names:
        oracle_start = time.perf_counter()
        print(f"\n{'=' * 62}\nORACLE = {oracle}\n{'=' * 62}")

        x_all, y_all = prepare_pool(
            dataset_mode, oracle, x_clean, y_clean, mean
        )
        N = len(x_all)

        print(
            f"Dataset pool: {dataset_mode} | "
            f"Samples: {N}"
        )

        # -------------------------------------------------
        # Predictions from all configured models
        # Measure each model separately so end-to-end timing can
        # include only the inference actually required by each method.
        # -------------------------------------------------
        model_inference_times = {}
        all_probs_list = []

        for n in names:
            t_pred = time.perf_counter()
            p = models[n].predict(
                x_all,
                batch_size=BATCH,
                verbose=0,
            )
            model_inference_times[n] = time.perf_counter() - t_pred
            all_probs_list.append(p)

        all_probs = np.stack(
            all_probs_list,
            axis=0,
        )

        oi = names.index(oracle)
        oracle_probs = all_probs[oi]

        pred = np.argmax(oracle_probs, axis=1)
        faulty = (pred != y_all).astype(int)
        print(f"Pool: {N} | Faulty: {faulty.sum()}")

        # -------------------------------------------------
        # Ensemble: exclusive OR inclusive
        # -------------------------------------------------
        if ensemble_setting == "exclusive":
            ens = np.delete(all_probs, oi, axis=0)
        elif ensemble_setting == "inclusive":
            ens = all_probs
        else:
            raise ValueError(
                "ensemble_setting must be 'exclusive' or 'inclusive'."
            )

        expected_size = len(names) - 1 if ensemble_setting == "exclusive" else len(names)
        assert ens.shape[0] == expected_size

        mp = np.mean(ens, axis=0)

        oracle_inference_time = model_inference_times[oracle]
        ensemble_names = [
            names[j]
            for j in range(len(names))
            if ensemble_setting == "inclusive" or j != oi
        ]
        ensemble_inference_time = sum(
            model_inference_times[n] for n in ensemble_names
        )

        # All models are needed only when a method explicitly requires
        # both the oracle and the ensemble containing/excluding it.
        adaptive_extra_oracle_time = (
            0.0 if ensemble_setting == "inclusive" else oracle_inference_time
        )

        rankings = {}

        # -------------------------------------------------
        # SINGLE-MODEL UNCERTAINTY
        # -------------------------------------------------
        t0 = time.perf_counter()
        gini = deepgini(oracle_probs)
        rankings["DeepGini"] = rank_desc(gini, rng_local)
        times["DeepGini"].append(time.perf_counter() - t0)
        end_to_end_times["DeepGini"].append(oracle_inference_time + times["DeepGini"][-1])

        t0 = time.perf_counter()
        ent = entropy(oracle_probs)
        rankings["Entropy"] = rank_desc(ent, rng_local)
        times["Entropy"].append(time.perf_counter() - t0)
        end_to_end_times["Entropy"].append(oracle_inference_time + times["Entropy"][-1])

        t0 = time.perf_counter()
        soft = softmax_confidence(oracle_probs)
        rankings["Softmax"] = rank_asc(soft, rng_local)
        times["Softmax"].append(time.perf_counter() - t0)
        end_to_end_times["Softmax"].append(oracle_inference_time + times["Softmax"][-1])

        t0 = time.perf_counter()
        pcs_s = pcs(oracle_probs)
        rankings["PCS"] = rank_asc(pcs_s, rng_local)
        times["PCS"].append(time.perf_counter() - t0)
        end_to_end_times["PCS"].append(oracle_inference_time + times["PCS"][-1])

        # -------------------------------------------------
        # ENSEMBLE UNCERTAINTY
        # -------------------------------------------------
        t0 = time.perf_counter()
        gini_ens = deepgini(mp)
        rankings["EnsGini"] = rank_desc(gini_ens, rng_local)
        times["EnsGini"].append(time.perf_counter() - t0)
        end_to_end_times["EnsGini"].append(ensemble_inference_time + times["EnsGini"][-1])

        t0 = time.perf_counter()
        var = ensemble_variance(ens)
        rankings["Variance"] = rank_desc(var, rng_local)
        times["Variance"].append(time.perf_counter() - t0)
        end_to_end_times["Variance"].append(ensemble_inference_time + times["Variance"][-1])

        t0 = time.perf_counter()
        dis = disagreement(mp)
        rankings["Disagreement"] = rank_desc(dis, rng_local)
        times["Disagreement"].append(time.perf_counter() - t0)
        end_to_end_times["Disagreement"].append(ensemble_inference_time + times["Disagreement"][-1])

        t0 = time.perf_counter()
        mi = mutual_information(ens)
        rankings["MutualInfo"] = rank_desc(mi, rng_local)
        times["MutualInfo"].append(time.perf_counter() - t0)
        end_to_end_times["MutualInfo"].append(ensemble_inference_time + times["MutualInfo"][-1])

        t0 = time.perf_counter()
        av = (1 - normalize(var)) * normalize(gini) + normalize(var) ** 2

        t0 = time.perf_counter()
        ami = (
            0.5 * normalize(entropy(mp))
            + 0.5 * normalize(mutual_information(ens))
        )
        rankings["AdaptiveMI"] = rank_desc(ami, rng_local)
        times["AdaptiveMI"].append(time.perf_counter() - t0)
        end_to_end_times["AdaptiveMI"].append(
            ensemble_inference_time
            + times["AdaptiveMI"][-1]
        )

        # -------------------------------------------------
        # ACTIVATIONS
        # -------------------------------------------------
        # Activation preparation time includes activation-model construction,
        # train-set profiling/statistics, and evaluation-pool extraction.
        t_ex0 = time.perf_counter()
        act_model = build_activation_model(models[oracle])
        A_train, _ = extract_activations(act_model, x_train)
        std_t = A_train.std(axis=0)
        max_t = A_train.max(axis=0)
        min_t = A_train.min(axis=0)

        A_all, layer_ids = extract_activations(act_model, x_all)
        t_ex = time.perf_counter() - t_ex0

        print(f"Neurons: {A_all.shape[1]} | extraction {t_ex:.1f}s")

        # -------------------------------------------------
        # CTM / CAM helper
        # -------------------------------------------------
        def do_ctm_cam(tag, cov, cov_compute_time):
            # CTM: activation preparation + coverage construction
            # + prioritization/ranking.
            t0 = time.perf_counter()
            rankings[f"{tag}-CTM"] = rank_desc(
                cov.sum(axis=1), rng_local
            )
            rank_time = time.perf_counter() - t0

            times[f"{tag}-CTM"].append(rank_time)
            end_to_end_times[f"{tag}-CTM"].append(
                t_ex + cov_compute_time + rank_time
            )

            # CAM: activation preparation + coverage construction
            # + greedy set-cover prioritization.
            t0 = time.perf_counter()
            r, n_p, cr = greedy_cam(
                cov, rng_local, CAM_MAX_PICKS
            )
            rank_time = time.perf_counter() - t0

            rankings[f"{tag}-CAM"] = r
            times[f"{tag}-CAM"].append(rank_time)
            end_to_end_times[f"{tag}-CAM"].append(
                t_ex + cov_compute_time + rank_time
            )

            maxcov[f"{tag}-CTM"].append(cr)
            maxcov[f"{tag}-CAM"].append(cr)
            picks[f"{tag}-CAM"].append(n_p)

            s = cov.sum(axis=1)
            print(
                f"  {tag:12s} maxcov={cr * 100:5.1f}%/{n_p:<6d} "
                f"nonzero={(s > 0).mean():.4f} "
                f"rand_tail={(N - n_p) / N:.4f}"
            )

        # -------------------------------------------------
        # COVERAGE METHODS
        # -------------------------------------------------
        for t in NAC_TS:
            t_cov0 = time.perf_counter()
            cov = nac_cover(A_all, t)
            cov_compute_time = time.perf_counter() - t_cov0
            do_ctm_cam(f"NAC({t})", cov, cov_compute_time)

        for s in NBC_STDS:
            t_cov0 = time.perf_counter()
            cov = nbc_cover(
                A_all,
                max_t + s * std_t,
                min_t - s * std_t,
            )
            cov_compute_time = time.perf_counter() - t_cov0
            do_ctm_cam(f"NBC({s})", cov, cov_compute_time)

        for s in SNAC_STDS:
            t_cov0 = time.perf_counter()
            cov = snac_cover(
                A_all,
                max_t + s * std_t,
            )
            cov_compute_time = time.perf_counter() - t_cov0
            do_ctm_cam(f"SNAC({s})", cov, cov_compute_time)

        for k in TKNC_KS:
            t_cov0 = time.perf_counter()
            cov = tknc_cover(A_all, layer_ids, k)
            cov_compute_time = time.perf_counter() - t_cov0

            t0 = time.perf_counter()
            r, n_p, cr = greedy_cam(
                cov, rng_local, CAM_MAX_PICKS
            )
            rank_time = time.perf_counter() - t0
            rankings[f"TKNC({k})-CAM"] = r
            times[f"TKNC({k})-CAM"].append(rank_time)
            end_to_end_times[f"TKNC({k})-CAM"].append(
                t_ex + cov_compute_time + rank_time
            )
            maxcov[f"TKNC({k})-CAM"].append(cr)
            picks[f"TKNC({k})-CAM"].append(n_p)
            print(
                f"  TKNC({k})     maxcov={cr * 100:5.1f}%/"
                f"{n_p:<6d} [CTM N/A]"
            )

        for b in KMNC_BINS_LIST:
            t_cov0 = time.perf_counter()
            bins = kmnc_bin_ids(
                A_all, min_t, max_t, b
            )
            cov_compute_time = time.perf_counter() - t_cov0

            t0 = time.perf_counter()
            r, n_p, cr = greedy_cam_kmnc(
                bins, b, rng_local, CAM_MAX_PICKS
            )
            rank_time = time.perf_counter() - t0
            rankings[f"KMNC({b})-CAM"] = r
            end_to_end_time = t_ex + cov_compute_time + rank_time
            times[f"KMNC({b})-CAM"].append(rank_time)
            end_to_end_times[f"KMNC({b})-CAM"].append(end_to_end_time)
            maxcov[f"KMNC({b})-CAM"].append(cr)
            picks[f"KMNC({b})-CAM"].append(n_p)
            print(
                f"  KMNC({b})  maxcov={cr * 100:5.1f}%/"
                f"{n_p:<6d} time={end_to_end_time:.1f}s [CTM N/A]"
            )

        # -------------------------------------------------
        # RANDOM
        # -------------------------------------------------
        t0 = time.perf_counter()
        rankings["Random"] = rng_local.permutation(N)
        times["Random"].append(time.perf_counter() - t0)
        end_to_end_times["Random"].append(times["Random"][-1])

        # -------------------------------------------------
        # APFD
        # -------------------------------------------------
        oracle_apfd = {}
        oracle_time = {}
        oracle_score_ranking_time = {}

        for metric in metric_names:
            apfd_value = APFD(rankings[metric], faulty)
            results[metric].append(apfd_value)
            oracle_apfd[metric] = float(apfd_value)
            oracle_time[metric] = float(end_to_end_times[metric][-1])
            oracle_score_ranking_time[metric] = float(times[metric][-1])

        per_oracle[oracle] = {
            "APFD": oracle_apfd,
            "prioritization_time_seconds": oracle_time,
            "score_ranking_time_seconds": oracle_score_ranking_time,
            "activation_extraction_seconds": float(t_ex),
            "pool_size": int(N),
            "faulty_count": int(faulty.sum()),
            "oracle_wall_clock_seconds": float(
                time.perf_counter() - oracle_start
            ),
        }

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    summary = {}
    for metric in metric_names:
        summary[metric] = {
            "APFD_mean": float(np.mean(results[metric])),
            "APFD_std": float(np.std(results[metric])),
            "prioritization_time_seconds_mean": float(
                np.mean(end_to_end_times[metric])
            ),
            "prioritization_time_seconds_std": float(
                np.std(end_to_end_times[metric])
            ),
            "score_ranking_time_seconds_mean": float(
                np.mean(times[metric])
            ),
            "score_ranking_time_seconds_std": float(
                np.std(times[metric])
            ),
        }
        if metric in maxcov:
            summary[metric]["max_coverage_mean"] = float(
                np.mean(maxcov[metric])
            )
        if metric in picks:
            summary[metric]["mean_CAM_picks"] = float(
                np.mean(picks[metric])
            )

    finished_at = utc_now()
    wall_clock = time.perf_counter() - run_start_wall

    output_json = os.path.join(
        BASE,
        f"cifar10_resnet20_{dataset_mode}_"
        f"apfd_time_by_oracle_{len(names)}models_"
        f"{ensemble_setting}.json",
    )

    payload = {
        "metadata": {
            "dataset": "CIFAR-10",
            "model_architecture": "ResNet-20",
            "model_names": names,
            "num_models": len(names),
            "ensemble_setting": ensemble_setting,
            "dataset_mode": dataset_mode,
            "random_seed": SEED,
            "batch_size": BATCH,
            "CAM_MAX_PICKS": CAM_MAX_PICKS,
            "timing_definition": (
                "End-to-end prioritization time per method, including "
                "method-required model inference, activation preparation/profiling "
                "for coverage methods, coverage/bin computation, score computation, "
                "and final ranking; model training excluded."
            ),
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "wall_clock_time_seconds": float(wall_clock),
            "wall_clock_time_minutes": float(wall_clock / 60.0),
        },
        "per_oracle": per_oracle,
        "summary": summary,
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 78)
    print(
        f"APFD | {dataset_mode.upper()} | "
        f"{ensemble_setting.upper()}"
    )
    print("=" * 78)
    for metric in metric_names:
        s = summary[metric]
        print(
            f"{metric:18s}: "
            f"{s['APFD_mean']:.4f} ± {s['APFD_std']:.4f} | "
            f"time={s['prioritization_time_seconds_mean']:.4f} ± "
            f"{s['prioritization_time_seconds_std']:.4f}s"
        )

    print(
        f"\nWall-clock: {wall_clock:.2f}s "
        f"({wall_clock / 60:.2f} min)"
    )
    print(f"Saved JSON: {output_json}")

    return payload, output_json

def generate_latex_table(all_payloads, output_tex):
    """Generate one LaTeX table containing all four experiments."""
    _, _, metric_names = build_metric_names()

    labels = [
        ("clean_exclusive", "Clean + Excl.", "clean", "exclusive"),
        ("clean_inclusive", "Clean + Incl.", "clean", "inclusive"),
        ("adversarial_exclusive", "Adv. + Excl.", "adversarial", "exclusive"),
        ("adversarial_inclusive", "Adv. + Incl.", "adversarial", "inclusive"),
    ]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Comparison of test case prioritization methods on CIFAR-10 "
        r"under clean and adversarial inputs with exclusive and inclusive "
        r"ensemble settings.}")
    lines.append(r"\label{tab:cifar10_four_settings}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Metric} & "
        r"\multicolumn{2}{c}{\textbf{Clean--Exclusive}} & "
        r"\multicolumn{2}{c}{\textbf{Clean--Inclusive}} & "
        r"\multicolumn{2}{c}{\textbf{Adversarial--Exclusive}} & "
        r"\multicolumn{2}{c}{\textbf{Adversarial--Inclusive}} \\"
    )
    lines.append(
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} "
        r"\cmidrule(lr){6-7} \cmidrule(lr){8-9}"
    )
    lines.append(
        r" & \textbf{APFD} & \textbf{Time (s)} "
        r"& \textbf{APFD} & \textbf{Time (s)} "
        r"& \textbf{APFD} & \textbf{Time (s)} "
        r"& \textbf{APFD} & \textbf{Time (s)} \\"
    )
    lines.append(r"\midrule")

    # Section classification follows the experiment structure.
    sections = [
        ("Coverage-Based Metrics", [
            k for k in metric_names
            if "-CTM" in k or "-CAM" in k
        ]),
        ("Single-Model Metrics", [
            "DeepGini", "Entropy", "Softmax", "PCS"
        ]),
        ("Ensemble-Based Metrics", [
            "EnsGini", "Variance", "Disagreement", "MutualInfo",
, "AdaptiveMI"
        ]),
        ("Baseline", ["Random"]),
    ]

    for section_name, metrics in sections:
        lines.append(
            f"\\multicolumn{{9}}{{c}}{{\\textbf{{{section_name}}}}} \\\\"
        )
        for metric in metrics:
            vals = []
            for key, _, _, _ in labels:
                s = all_payloads[key]["summary"][metric]
                vals.append(
                    f"{s['APFD_mean']:.4f} $\\pm$ {s['APFD_std']:.4f}"
                )
                vals.append(
                    f"{s['prioritization_time_seconds_mean']:.4f} "
                    f"$\\pm$ {s['prioritization_time_seconds_std']:.4f}"
                )
            lines.append(
                metric + " & " + " & ".join(vals) + r" \\"
            )
        if section_name != "Baseline":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    Path(output_tex).write_text("\n".join(lines) + "\n")

def main():
    overall_start = time.perf_counter()

    print("\nCIFAR-10 ResNet-20 | FOUR EXPERIMENTS\n")

    # Load data once.
    (x_train, y_train), (x_clean, y_clean), mean = load_cifar()

    # Load all models once and reuse them across the four runs.
    models = {
        k: tf.keras.models.load_model(v, compile=False)
        for k, v in MODELS.items()
    }
    names = list(models.keys())

    print(f"Models: {names}")
    print(f"Number of models: {len(names)}")
    print("Runs: clean/exclusive, clean/inclusive, adversarial/exclusive, "
          "adversarial/inclusive")

    all_payloads = {}

    for dataset_mode, ensemble_setting in RUN_CONFIGS:
        key = f"{dataset_mode}_{ensemble_setting}"
        payload, _ = run_one_configuration(
            dataset_mode=dataset_mode,
            ensemble_setting=ensemble_setting,
            models=models,
            names=names,
            x_train=x_train,
            x_clean=x_clean,
            y_clean=y_clean,
            mean=mean,
        )
        all_payloads[key] = payload

    # -----------------------------------------------------
    # Generate one LaTeX table automatically.
    # -----------------------------------------------------
    table_tex = os.path.join(
        BASE,
        f"cifar10_resnet20_four_settings_table_{len(names)}models.tex",
    )
    generate_latex_table(all_payloads, table_tex)

    # Also save a combined machine-readable JSON containing all four runs.
    combined_json = os.path.join(
        BASE,
        f"cifar10_resnet20_four_settings_{len(names)}models.json",
    )
    combined_payload = {
        "metadata": {
            "dataset": "CIFAR-10",
            "model_architecture": "ResNet-20",
            "model_names": names,
            "num_models": len(names),
            "random_seed": SEED,
            "batch_size": BATCH,
            "run_order": [
                "clean_exclusive",
                "clean_inclusive",
                "adversarial_exclusive",
                "adversarial_inclusive",
            ],
        },
        "runs": all_payloads,
    }
    with open(combined_json, "w") as f:
        json.dump(combined_payload, f, indent=2)

    overall_time = time.perf_counter() - overall_start

    print("\n" + "=" * 78)
    print("====== ALL FOUR EXPERIMENTS COMPLETED ======")
    print("=" * 78)
    print(f"Four-run wall-clock time: {overall_time:.2f} seconds "
          f"({overall_time / 60:.2f} minutes)")
    print(f"Combined JSON: {combined_json}")
    print(f"LaTeX table:   {table_tex}")
    print("=" * 78)

if __name__ == "__main__":
    main()