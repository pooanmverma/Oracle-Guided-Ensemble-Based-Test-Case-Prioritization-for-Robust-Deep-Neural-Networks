"""
MNIST (LeNet-1) rotating-oracle TCP comparison:
    uncertainty / ensemble metrics  vs  coverage-based baselines

END-TO-END TIMING PORTED FROM THE CIFAR-10 SCRIPT (consistent across datasets):
  * time.perf_counter() everywhere
  * per-model inference timed separately (model_inference_times)
  * each method's end_to_end_time INCLUDES the inference it actually needs:
      single-model  -> oracle inference
      ensemble      -> ensemble inference (incl/excl oracle per setting)
      AdaptiveMI    -> ensemble inference
      coverage      -> activation extraction (t_ex) + coverage build + ranking
      Random        -> ranking only (no inference)
  * separate `times` (score+ranking) and `end_to_end_times` (inference-inclusive)
  * AdaptiveMI uses entropy(mean_probs) [ensemble-mean entropy] to match CIFAR,
    so it is a purely ensemble-based metric (NOTE: Fashion/older MNIST used
    entropy(oracle_probs); this file now matches CIFAR).

Output JSON schema MATCHES the CIFAR four-settings file (runs -> {metadata,
per_oracle, summary}), so the same table generator works for all datasets.

Runs all four configurations: clean/adversarial x exclusive/inclusive.
"""

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

BATCH = 256
SEED = 42
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

MODEL_SEEDS = [0, 1, 2, 3, 4]
MODELS = {f"seed{s}": f"{BASE}/models/model_mnist_seed{s}.h5" for s in MODEL_SEEDS}
ADV_DIR = f"{BASE}/adv_data"

NAC_TS = [0.0, 0.75]
NBC_STDS = [0.0, 0.5, 1.0]
SNAC_STDS = [0.0, 0.5, 1.0]
TKNC_KS = [1, 2, 3]
KMNC_BINS_LIST = [1000, 10000]
CAM_MAX_PICKS = None

RUN_CONFIGS = [
    ("clean", "exclusive"),
    ("clean", "inclusive"),
    ("adversarial", "exclusive"),
    ("adversarial", "inclusive"),
]

# =====================================================
# DATA
# =====================================================

def load_mnist():
    (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
    xtr = np.expand_dims(xtr.astype("float32") / 255.0, -1)
    xte = np.expand_dims(xte.astype("float32") / 255.0, -1)
    return (xtr, ytr.astype("int64")), (xte, yte.astype("int64"))

def load_adv_for(seed):
    """Model-specific FGSM and BIM adversarial samples (each 10,000)."""
    fgsm_path = os.path.join(ADV_DIR, f"mnist_fgsm_seed{seed}.npy")
    bim_path = os.path.join(ADV_DIR, f"mnist_bim_seed{seed}.npy")
    if not os.path.isfile(fgsm_path):
        raise FileNotFoundError(f"Missing FGSM adversarial file: {fgsm_path}")
    if not os.path.isfile(bim_path):
        raise FileNotFoundError(f"Missing BIM adversarial file: {bim_path}")
    return np.load(fgsm_path).astype("float32"), np.load(bim_path).astype("float32")

def prepare_pool(dataset_mode, oracle, x_clean, y_clean):
    if dataset_mode == "clean":
        return x_clean, y_clean
    if dataset_mode != "adversarial":
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")
    seed = int(oracle.replace("seed", ""))
    x_fgsm, x_bim = load_adv_for(seed)
    if len(x_fgsm) != len(x_clean) or len(x_bim) != len(x_clean):
        raise ValueError(
            f"Adversarial size mismatch for {oracle}: "
            f"fgsm={len(x_fgsm)} bim={len(x_bim)} clean={len(x_clean)}"
        )
    x_all = np.concatenate([x_clean, x_fgsm, x_bim], axis=0)
    y_all = np.tile(y_clean, 3)
    return x_all, y_all

# =====================================================
# ACTIVATION EXTRACTION  (LeNet-1: Conv/Dense + separate Activation)
# =====================================================

def build_activation_model(model):
    layers = model.layers
    outs = []
    for i, layer in enumerate(layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            if isinstance(nxt, tf.keras.layers.Activation):
                outs.append(nxt.output)
            else:
                outs.append(layer.output)
    if len(outs) > 1:
        outs = outs[:-1]   # drop final (softmax) output
    return tf.keras.models.Model(inputs=model.inputs, outputs=outs)

def extract_activations(act_model, x, batch=BATCH):
    outs = act_model.predict(x, batch_size=batch, verbose=0)
    if not isinstance(outs, list):
        outs = [outs]
    feats, layer_ids = [], []
    for i, o in enumerate(outs):
        if o.ndim == 4:
            o = o.mean(axis=(1, 2))
        feats.append(o.astype("float32"))
        layer_ids.append(np.full(o.shape[1], i, dtype=np.int32))
    return np.concatenate(feats, axis=1), np.concatenate(layer_ids)

# =====================================================
# RANKING / COVERAGE HELPERS
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

def nac_cover(A, t): return (1.0 / (1.0 + np.exp(-A))) > t
def snac_cover(A, upper): return A > upper
def nbc_cover(A, upper, lower): return np.concatenate([A > upper, A < lower], axis=1)

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

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def build_metric_names():
    unc_names = ["DeepGini", "Entropy", "Softmax", "PCS",
                 "EnsGini", "Variance", "Disagreement", "MutualInfo",
, "AdaptiveMI"]
    cov_names = []
    for t in NAC_TS:   cov_names += [f"NAC({t})-CTM", f"NAC({t})-CAM"]
    for s in NBC_STDS: cov_names += [f"NBC({s})-CTM", f"NBC({s})-CAM"]
    for s in SNAC_STDS: cov_names += [f"SNAC({s})-CTM", f"SNAC({s})-CAM"]
    for k in TKNC_KS:  cov_names.append(f"TKNC({k})-CAM")
    for b in KMNC_BINS_LIST: cov_names.append(f"KMNC({b})-CAM")
    return unc_names, cov_names, unc_names + cov_names + ["Random"]

# =====================================================
# ONE CONFIGURATION
# =====================================================

def run_one_configuration(dataset_mode, ensemble_setting, models, names,
                          x_train, x_clean, y_clean):
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
    print(f"RUN: {dataset_mode.upper()} | {ensemble_setting.upper()} | "
          f"{len(names)}-MODEL ENSEMBLE")
    print("#" * 78)

    for oracle in names:
        oracle_start = time.perf_counter()
        print(f"\n{'=' * 62}\nORACLE = {oracle}\n{'=' * 62}")

        x_all, y_all = prepare_pool(dataset_mode, oracle, x_clean, y_clean)
        N = len(x_all)
        print(f"Dataset pool: {dataset_mode} | Samples: {N}")

        # ---- per-model inference, timed separately ----
        model_inference_times = {}
        all_probs_list = []
        for n in names:
            t_pred = time.perf_counter()
            p = models[n].predict(x_all, batch_size=BATCH, verbose=0)
            model_inference_times[n] = time.perf_counter() - t_pred
            all_probs_list.append(p)
        all_probs = np.stack(all_probs_list, axis=0)

        oi = names.index(oracle)
        oracle_probs = all_probs[oi]
        pred = np.argmax(oracle_probs, axis=1)
        faulty = (pred != y_all).astype(int)
        print(f"Pool: {N} | Faulty: {faulty.sum()}")

        if ensemble_setting == "exclusive":
            ens = np.delete(all_probs, oi, axis=0)
        elif ensemble_setting == "inclusive":
            ens = all_probs
        else:
            raise ValueError("ensemble_setting must be 'exclusive' or 'inclusive'.")
        expected_size = len(names) - 1 if ensemble_setting == "exclusive" else len(names)
        assert ens.shape[0] == expected_size
        mp = np.mean(ens, axis=0)

        oracle_inference_time = model_inference_times[oracle]
        ensemble_names = [names[j] for j in range(len(names))
                          if ensemble_setting == "inclusive" or j != oi]
        ensemble_inference_time = sum(model_inference_times[n] for n in ensemble_names)
        # is already inside the ensemble; in exclusive add it once.
        adaptive_extra_oracle_time = (
            0.0 if ensemble_setting == "inclusive" else oracle_inference_time
        )

        rankings = {}

        # ---- SINGLE-MODEL UNCERTAINTY (end_to_end += oracle inference) ----
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

        # ---- ENSEMBLE UNCERTAINTY (end_to_end += ensemble inference) ----
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
        # matches CIFAR: ensemble-mean entropy (purely ensemble-based)
        ami = 0.5 * normalize(entropy(mp)) + 0.5 * normalize(mutual_information(ens))
        rankings["AdaptiveMI"] = rank_desc(ami, rng_local)
        times["AdaptiveMI"].append(time.perf_counter() - t0)
        end_to_end_times["AdaptiveMI"].append(ensemble_inference_time + times["AdaptiveMI"][-1])

        # ---- ACTIVATIONS (t_ex = construction + train profiling + pool extraction) ----
        t_ex0 = time.perf_counter()
        act_model = build_activation_model(models[oracle])
        A_train, _ = extract_activations(act_model, x_train)
        std_t = A_train.std(axis=0); max_t = A_train.max(axis=0); min_t = A_train.min(axis=0)
        A_all, layer_ids = extract_activations(act_model, x_all)
        t_ex = time.perf_counter() - t_ex0
        print(f"Neurons: {A_all.shape[1]} | extraction {t_ex:.1f}s")

        def do_ctm_cam(tag, cov, cov_compute_time):
            t0 = time.perf_counter()
            rankings[f"{tag}-CTM"] = rank_desc(cov.sum(axis=1), rng_local)
            rank_time = time.perf_counter() - t0
            times[f"{tag}-CTM"].append(rank_time)
            end_to_end_times[f"{tag}-CTM"].append(t_ex + cov_compute_time + rank_time)

            t0 = time.perf_counter()
            r, n_p, cr = greedy_cam(cov, rng_local, CAM_MAX_PICKS)
            rank_time = time.perf_counter() - t0
            rankings[f"{tag}-CAM"] = r
            times[f"{tag}-CAM"].append(rank_time)
            end_to_end_times[f"{tag}-CAM"].append(t_ex + cov_compute_time + rank_time)

            maxcov[f"{tag}-CTM"].append(cr); maxcov[f"{tag}-CAM"].append(cr)
            picks[f"{tag}-CAM"].append(n_p)

        for t in NAC_TS:
            t_c = time.perf_counter(); cov = nac_cover(A_all, t); ct = time.perf_counter() - t_c
            do_ctm_cam(f"NAC({t})", cov, ct)
        for s in NBC_STDS:
            t_c = time.perf_counter(); cov = nbc_cover(A_all, max_t + s * std_t, min_t - s * std_t); ct = time.perf_counter() - t_c
            do_ctm_cam(f"NBC({s})", cov, ct)
        for s in SNAC_STDS:
            t_c = time.perf_counter(); cov = snac_cover(A_all, max_t + s * std_t); ct = time.perf_counter() - t_c
            do_ctm_cam(f"SNAC({s})", cov, ct)
        for k in TKNC_KS:
            t_c = time.perf_counter(); cov = tknc_cover(A_all, layer_ids, k); ct = time.perf_counter() - t_c
            t0 = time.perf_counter()
            r, n_p, cr = greedy_cam(cov, rng_local, CAM_MAX_PICKS)
            rank_time = time.perf_counter() - t0
            rankings[f"TKNC({k})-CAM"] = r
            times[f"TKNC({k})-CAM"].append(rank_time)
            end_to_end_times[f"TKNC({k})-CAM"].append(t_ex + ct + rank_time)
            maxcov[f"TKNC({k})-CAM"].append(cr); picks[f"TKNC({k})-CAM"].append(n_p)
        for b in KMNC_BINS_LIST:
            t_c = time.perf_counter(); bins = kmnc_bin_ids(A_all, min_t, max_t, b); ct = time.perf_counter() - t_c
            t0 = time.perf_counter()
            r, n_p, cr = greedy_cam_kmnc(bins, b, rng_local, CAM_MAX_PICKS)
            rank_time = time.perf_counter() - t0
            rankings[f"KMNC({b})-CAM"] = r
            times[f"KMNC({b})-CAM"].append(rank_time)
            end_to_end_times[f"KMNC({b})-CAM"].append(t_ex + ct + rank_time)
            maxcov[f"KMNC({b})-CAM"].append(cr); picks[f"KMNC({b})-CAM"].append(n_p)

        # ---- RANDOM (no inference) ----
        t0 = time.perf_counter()
        rankings["Random"] = rng_local.permutation(N)
        times["Random"].append(time.perf_counter() - t0)
        end_to_end_times["Random"].append(times["Random"][-1])

        # ---- APFD ----
        oracle_apfd, oracle_time, oracle_score_time = {}, {}, {}
        for metric in metric_names:
            a = APFD(rankings[metric], faulty)
            results[metric].append(a)
            oracle_apfd[metric] = float(a)
            oracle_time[metric] = float(end_to_end_times[metric][-1])
            oracle_score_time[metric] = float(times[metric][-1])

        per_oracle[oracle] = {
            "APFD": oracle_apfd,
            "prioritization_time_seconds": oracle_time,
            "score_ranking_time_seconds": oracle_score_time,
            "activation_extraction_seconds": float(t_ex),
            "pool_size": int(N),
            "faulty_count": int(faulty.sum()),
            "oracle_wall_clock_seconds": float(time.perf_counter() - oracle_start),
        }

    # ---- summary ----
    summary = {}
    for metric in metric_names:
        summary[metric] = {
            "APFD_mean": float(np.mean(results[metric])),
            "APFD_std": float(np.std(results[metric])),
            "prioritization_time_seconds_mean": float(np.mean(end_to_end_times[metric])),
            "prioritization_time_seconds_std": float(np.std(end_to_end_times[metric])),
            "score_ranking_time_seconds_mean": float(np.mean(times[metric])),
            "score_ranking_time_seconds_std": float(np.std(times[metric])),
        }
        if metric in maxcov:
            summary[metric]["max_coverage_mean"] = float(np.mean(maxcov[metric]))
        if metric in picks:
            summary[metric]["mean_CAM_picks"] = float(np.mean(picks[metric]))

    wall_clock = time.perf_counter() - run_start_wall
    output_json = os.path.join(
        BASE, f"mnist_lenet1_{dataset_mode}_apfd_time_by_oracle_"
              f"{len(names)}models_{ensemble_setting}.json")
    payload = {
        "metadata": {
            "dataset": "MNIST", "model_architecture": "LeNet-1",
            "model_names": names, "num_models": len(names),
            "ensemble_setting": ensemble_setting, "dataset_mode": dataset_mode,
            "random_seed": SEED, "batch_size": BATCH, "CAM_MAX_PICKS": CAM_MAX_PICKS,
            "timing_definition": (
                "End-to-end prioritization time per method, including method-required "
                "model inference, activation preparation/profiling for coverage methods, "
                "coverage/bin computation, score computation, and final ranking; "
                "model training excluded."),
            "started_at_utc": started_at, "finished_at_utc": utc_now(),
            "wall_clock_time_seconds": float(wall_clock),
            "wall_clock_time_minutes": float(wall_clock / 60.0),
        },
        "per_oracle": per_oracle, "summary": summary,
    }
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 78)
    print(f"APFD | {dataset_mode.upper()} | {ensemble_setting.upper()}")
    print("=" * 78)
    for metric in metric_names:
        s = summary[metric]
        print(f"{metric:18s}: {s['APFD_mean']:.4f} ± {s['APFD_std']:.4f} | "
              f"time={s['prioritization_time_seconds_mean']:.4f} ± "
              f"{s['prioritization_time_seconds_std']:.4f}s")
    print(f"\nWall-clock: {wall_clock:.2f}s ({wall_clock/60:.2f} min)")
    print(f"Saved JSON: {output_json}")
    return payload, output_json

# =====================================================
# MAIN
# =====================================================

def main():
    overall_start = time.perf_counter()
    print("\nMNIST LeNet-1 | FOUR EXPERIMENTS\n")

    (x_train, y_train), (x_clean, y_clean) = load_mnist()
    models = {k: tf.keras.models.load_model(v, compile=False) for k, v in MODELS.items()}
    names = list(models.keys())
    print(f"Models: {names}")

    all_payloads = {}
    for dataset_mode, ensemble_setting in RUN_CONFIGS:
        key = f"{dataset_mode}_{ensemble_setting}"
        payload, _ = run_one_configuration(
            dataset_mode, ensemble_setting, models, names, x_train, x_clean, y_clean)
        all_payloads[key] = payload

    combined_json = os.path.join(BASE, f"mnist_lenet1_four_settings_{len(names)}models.json")
    combined_payload = {
        "metadata": {
            "dataset": "MNIST", "model_architecture": "LeNet-1",
            "model_names": names, "num_models": len(names),
            "random_seed": SEED, "batch_size": BATCH,
            "run_order": ["clean_exclusive", "clean_inclusive",
                          "adversarial_exclusive", "adversarial_inclusive"],
        },
        "runs": all_payloads,
    }
    with open(combined_json, "w") as f:
        json.dump(combined_payload, f, indent=2)

    overall_time = time.perf_counter() - overall_start
    print("\n" + "=" * 78)
    print("====== ALL FOUR MNIST EXPERIMENTS COMPLETED ======")
    print("=" * 78)
    print(f"Four-run wall-clock: {overall_time:.2f}s ({overall_time/60:.2f} min)")
    print(f"Combined JSON: {combined_json}")
    print("=" * 78)

if __name__ == "__main__":
    main()