"""
Statistical tests on 10-model Fashion-MNIST (LeNet-5).

With n=10 oracle rotations:
  - Exact permutation test: 2^10 = 1024 permutations → min p = 2/1024 ≈ 0.00195
  - Wilcoxon signed-rank test: n≥6 → p<0.05 ACHIEVABLE
  - Cohen's d_z: paired effect size

Protocol: same rotating-oracle as the paper.
  For each oracle i (i=0..9):
    - Pool = adv(oracle_i) + clean   (MIX_CLEAN=True)
    - Faults = oracle_i misclassifies
    - Exclusive = ensemble of OTHER 9 models
    - Inclusive = ensemble of ALL 10 models
    - APFD for each metric in both settings
  Then: paired difference = inclusive_APFD - exclusive_APFD for each oracle round.
  Statistical tests on these 10 paired differences.

SAFETY: models loaded READ-ONLY (compile=False). Never .save().

Usage:  python stat_test_fashion10.py
"""

import os, gc, itertools
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from scipy import stats as sp_stats

# ============================================================ CONFIG
MIX_CLEAN = True
SEEDS = list(range(10))  # 0..9
ATTACKS = ["fgsm", "bim"]

# >>> PATHS — adjust if needed <<<
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = f"{BASE}/models"
ADV_DIR   = f"{BASE}/adv_data_fashion"   # check: might be adv_data or adv_data_fashion
ADV_PREFIX = "fashion"

# model paths
MODEL_PATHS = {s: f"{MODEL_DIR}/fashion_seed{s}.h5" for s in SEEDS}
# adv key per seed
ADV_KEY = {s: f"seed{s}" for s in SEEDS}

# ============================================================ METRICS
EPS = 1e-12

def deepgini(p):    return 1 - np.sum(p**2, axis=1)
def entropy(p):     return -(p * np.log(p + EPS)).sum(axis=1)
def conf(p):        return np.max(p, axis=1)
def pcs(p):
    sp = -np.sort(-p, axis=1); return sp[:,0] - sp[:,1]
def ens_var(pr):    return np.mean(np.var(pr, axis=0), axis=1)
def disagree(mp):   return 1 - np.max(mp, axis=1)
def mutual_info(ens):
    mp = ens.mean(0)
    hm = -(mp * np.log(mp + EPS)).sum(axis=1)
    he = -(ens * np.log(ens + EPS)).sum(axis=2)
    return hm - he.mean(axis=0)
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + EPS)

ENSEMBLE_METRICS = ["EnsGini", "Variance", "Disagreement", "MutualInfo", "AdaptiveMI"]

def compute_ensemble_scores(P_pool, oracle_idx, setting):
    n_models = len(P_pool)
    if setting == "exclusive":
        ens = P_pool[np.arange(n_models) != oracle_idx]
    else:
        ens = P_pool
    mp = ens.mean(0)
    ent_mean = entropy(mp)
    mi = mutual_info(ens)
    ami = 0.5 * normalize(ent_mean) + 0.5 * normalize(mi)
    return {
        "EnsGini":       deepgini(mp),
        "Variance":      ens_var(ens),
        "Disagreement":  disagree(mp),
        "MutualInfo":    mi,
        "AdaptiveMI":    ami,
    }

# ============================================================ APFD
def apfd(scores, faults):
    n = len(scores); m = int(faults.sum())
    if m == 0: return np.nan
    order = np.argsort(-scores, kind="stable")
    pos = np.empty(n, dtype=np.int64); pos[order] = np.arange(1, n+1)
    return 1.0 - pos[faults].sum() / (n * m) + 1.0 / (2 * n)

# ============================================================ STAT TESTS
def exact_permutation_test_twosided(diffs):
    """Exact two-sided permutation test on paired differences."""
    n = len(diffs)
    t_obs = abs(np.mean(diffs))
    count = 0
    total = 2**n
    for signs in itertools.product([-1, 1], repeat=n):
        t_perm = abs(np.mean(np.array(signs) * diffs))
        if t_perm >= t_obs - 1e-15:
            count += 1
    return count / total

def cohens_dz(diffs):
    """Paired Cohen's d_z = mean(diff) / std(diff)."""
    m = np.mean(diffs)
    s = np.std(diffs, ddof=1)
    if s < 1e-15: return np.inf if abs(m) > 0 else 0.0
    return m / s

# ============================================================ MAIN
def main():
    print("=" * 80)
    print("STATISTICAL TESTS — 10-Model Fashion-MNIST / LeNet-5")
    print("  n=10 oracle rotations → Wilcoxon feasible at α=0.05")
    print("=" * 80)

    # Check models exist
    for s, p in MODEL_PATHS.items():
        if not os.path.isfile(p):
            print(f"ERROR: model not found: {p}")
            return
    print("All 10 model files found.")

    # Load clean data
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_clean = np.expand_dims(x_test.astype("float32") / 255.0, -1)
    y_clean = y_test.flatten().astype("int64")
    print(f"Clean test set: {x_clean.shape[0]} samples")

    # Load models
    models = {}
    for s in SEEDS:
        models[s] = keras.models.load_model(MODEL_PATHS[s], compile=False)
        acc = (models[s].predict(x_clean, batch_size=256, verbose=0).argmax(1) == y_clean).mean()
        print(f"  seed{s} accuracy: {acc:.4f}")

    # Clean probs [10, N, C]
    P_clean = np.stack([models[s].predict(x_clean, batch_size=256, verbose=0) for s in SEEDS], axis=0)

    # results[attack][metric] = {'excl': [10 APFDs], 'incl': [10 APFDs]}
    results = {a: {m: {"excl": [], "incl": []} for m in ENSEMBLE_METRICS} for a in ATTACKS}

    for attack in ATTACKS:
        print(f"\n--- {attack.upper()} ---")
        for oi in SEEDS:
            adv_path = f"{ADV_DIR}/{ADV_PREFIX}_{attack}_{ADV_KEY[oi]}.npy"
            if not os.path.isfile(adv_path):
                print(f"  [SKIP] oracle seed{oi}: {adv_path} not found")
                continue

            x_adv = np.load(adv_path).astype("float32")
            if x_adv.max() > 1.5:
                x_adv = x_adv / 255.0
            if x_adv.ndim == 3:
                x_adv = np.expand_dims(x_adv, -1)

            # Probs on adv [10, N_adv, C]
            P_adv = np.stack([models[s].predict(x_adv, batch_size=256, verbose=0) for s in SEEDS], axis=0)

            if MIX_CLEAN:
                P_pool = np.concatenate([P_adv, P_clean], axis=1)
                y_pool = np.concatenate([y_clean, y_clean])
            else:
                P_pool = P_adv
                y_pool = y_clean

            faults = (P_pool[oi].argmax(1) != y_pool)
            fr = faults.mean()
            print(f"  Oracle seed{oi}: pool={len(y_pool)}, fault_rate={fr:.3f}")

            for setting, key in [("exclusive", "excl"), ("inclusive", "incl")]:
                scores = compute_ensemble_scores(P_pool, oi, setting)
                for m in ENSEMBLE_METRICS:
                    a_val = apfd(scores[m], faults)
                    results[attack][m][key].append(a_val)

    # Cleanup
    del models; tf.keras.backend.clear_session(); gc.collect()

    # ============================================================ STAT OUTPUT
    print(f"\n\n{'='*100}")
    print("STATISTICAL ANALYSIS — Inclusive vs Exclusive (paired, n=10)")
    print(f"{'='*100}")
    print(f"{'Attack':6s} {'Metric':<14s} {'Excl APFD':>10s} {'Incl APFD':>10s} "
          f"{'Gain(pp)':>9s} {'Perm-p':>8s} {'Wilcox-p':>9s} {'Cohen dz':>9s} {'Sig?':>5s}")
    print("-" * 100)

    alpha = 0.05
    all_rows = []

    for attack in ATTACKS:
        for m in ENSEMBLE_METRICS:
            excl = np.array(results[attack][m]["excl"])
            incl = np.array(results[attack][m]["incl"])
            if len(excl) < 10 or len(incl) < 10:
                print(f"{attack.upper():6s} {m:<14s}  SKIPPED (not enough data)")
                continue

            diffs = incl - excl
            e_mean = excl.mean()
            i_mean = incl.mean()
            gain = (i_mean - e_mean) * 100

            # Exact permutation test (2^10 = 1024)
            p_perm = exact_permutation_test_twosided(diffs)

            # Wilcoxon signed-rank test
            try:
                stat_w, p_wilcox = sp_stats.wilcoxon(diffs, alternative='two-sided')
            except Exception:
                p_wilcox = np.nan

            # Cohen's d_z
            dz = cohens_dz(diffs)

            sig = "YES" if (p_perm < alpha or p_wilcox < alpha) else "no"
            all_rows.append((attack, m, e_mean, i_mean, gain, p_perm, p_wilcox, dz, sig))

            print(f"{attack.upper():6s} {m:<14s} {e_mean:10.4f} {i_mean:10.4f} "
                  f"{gain:>+9.2f} {p_perm:8.4f} {p_wilcox:9.4f} {dz:9.2f} {sig:>5s}")
        print()

    # ============================================================ HOLM-BONFERRONI
    print(f"\n{'='*100}")
    print("HOLM-BONFERRONI CORRECTION (across all 10 comparisons)")
    print(f"{'='*100}")

    # Collect all p-values (permutation)
    pvals_perm = [(r[0], r[1], r[5]) for r in all_rows]  # attack, metric, p_perm
    pvals_wilc = [(r[0], r[1], r[6]) for r in all_rows]  # attack, metric, p_wilcox

    for label, pvals in [("Permutation", pvals_perm), ("Wilcoxon", pvals_wilc)]:
        print(f"\n  --- {label} p-values ---")
        sorted_p = sorted(pvals, key=lambda x: x[2])
        k = len(sorted_p)
        print(f"  {'Rank':>4s} {'Attack':6s} {'Metric':<14s} {'Raw p':>8s} {'Threshold':>10s} {'Holm Sig?':>10s}")
        any_fail = False
        for i, (att, met, p) in enumerate(sorted_p):
            threshold = alpha / (k - i)
            if not any_fail and p <= threshold:
                holm_sig = "YES"
            else:
                holm_sig = "no"
                any_fail = True
            print(f"  {i+1:>4d} {att.upper():6s} {met:<14s} {p:8.4f} {threshold:10.4f} {holm_sig:>10s}")

    # ============================================================ RAW DIFFS
    print(f"\n{'='*100}")
    print("RAW PER-ORACLE DIFFERENCES (Inclusive - Exclusive APFD)")
    print(f"{'='*100}")
    for attack in ATTACKS:
        for m in ENSEMBLE_METRICS:
            excl = np.array(results[attack][m]["excl"])
            incl = np.array(results[attack][m]["incl"])
            if len(excl) < 10: continue
            diffs = incl - excl
            signs = ["+" if d > 0 else ("-" if d < 0 else "0") for d in diffs]
            print(f"{attack.upper():6s} {m:<14s}: "
                  + " ".join(f"{d:+.4f}" for d in diffs)
                  + f"  [{'/'.join(signs)}]  all_pos={all(d>0 for d in diffs)}")

    print("\nDone.")


if __name__ == "__main__":
    main()