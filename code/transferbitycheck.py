"""
Exclusive vs Inclusive APFD under per-model adversarial attacks — ALL DATASETS.

Extends the original CIFAR-10-only script to all 6 datasets and all metrics
(single-model + ensemble). FGSM and BIM reported separately.

Protocol (rotating oracle — UNCHANGED from the paper):
  For each oracle i (each of the 5 models):
    - Test pool          = adversarial examples generated against oracle i
                           (+ clean originals, see MIX_CLEAN below)
    - Fault labels        = oracle i misclassifies the input (vs ground truth)
    - EXCLUSIVE uncertainty = aggregated over the OTHER 4 models (oracle held out)
    - INCLUSIVE uncertainty = aggregated over ALL 5 models (oracle kept in)
  Rank the SAME pool by uncertainty in each setting -> APFD.
  Average APFD over the 5 oracle rounds. Report FGSM and BIM separately.

SAFETY: models are loaded READ-ONLY (compile=False). Never call .save().

Usage:  python exclusive_vs_inclusive_apfd_ALL.py
"""

import os, gc
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras

# ============================================================ CONFIG
MIX_CLEAN = True   # True = pool is adv+clean; False = adv only (degenerate)

BASE = os.path.dirname(os.path.abspath(__file__))
SVHN_MAT  = f"{BASE}/data/test_32x32.mat"
SEEDS     = [0, 1, 2, 3, 4]
ATTACKS   = ["fgsm", "bim"]

_CIFAR_MEAN = None  # set in load_clean for cifar

# ============================================================ DATASETS
DATASETS = ["cifar", "mnist5", "mnist1", "svhn", "fashion", "fashion20"]

def get_config(ds):
    if ds == "cifar":
        C = {0:"resnet20_best", 1:"resnet20_seed1", 2:"resnet20_seed2",
             3:"resnet20_seed9", 4:"resnet20_seed23"}
        return dict(
            models={f"seed{s}": f"{BASE}/models/{C[s]}.h5" for s in SEEDS},
            adv_dir=f"{BASE}/adv_data", adv_prefix="cifar10",
            adv_key={s: C[s] for s in SEEDS},
            row="CIFAR-10 / ResNet-20")
    if ds == "mnist5":
        return dict(
            models={f"seed{s}": f"{BASE}/models/lenet1_mnist_adapted_seed{s}.h5" for s in SEEDS},
            adv_dir=f"{BASE}/adv_data", adv_prefix="mnist_lenet1",
            adv_key={s: f"seed{s}" for s in SEEDS},
            row="MNIST / LeNet-5")
    if ds == "mnist1":
        return dict(
            models={f"seed{s}": f"{BASE}/models/model_mnist_seed{s}.h5" for s in SEEDS},
            adv_dir=f"{BASE}/adv_data", adv_prefix="mnist",
            adv_key={s: f"seed{s}" for s in SEEDS},
            row="MNIST / LeNet-1")
    if ds == "svhn":
        return dict(
            models={f"seed{s}": f"{BASE}/model/svhn_seed{s}.h5" for s in SEEDS},
            adv_dir=f"{BASE}/adv_data_svhn", adv_prefix="svhn",
            adv_key={s: str(s) for s in SEEDS},
            row="SVHN / LeNet-5")
    if ds == "fashion":
        return dict(
            models={f"seed{s}": f"{BASE}/models/fashion_seed{s}.h5" for s in SEEDS},
            adv_dir=f"{BASE}/adv_data_fashion", adv_prefix="fashion",
            adv_key={s: f"seed{s}" for s in SEEDS},
            row="Fashion-MNIST / LeNet-5")
    if ds == "fashion20":
        return dict(
            models={f"seed{s}": f"{BASE}/models/resnet20_fmnist_seed{s}.h5" for s in SEEDS},
            adv_dir=f"{BASE}/adv_data_fashion20", adv_prefix="fashion",
            adv_key={s: f"seed{s}" for s in SEEDS},
            row="Fashion-MNIST / ResNet-20")
    raise ValueError(ds)

# ============================================================ DATA LOADING
def load_clean(ds):
    global _CIFAR_MEAN
    if ds in ("fashion", "fashion20"):
        (_, _), (x, y) = keras.datasets.fashion_mnist.load_data()
        return np.expand_dims(x.astype("float32")/255.0, -1), y.flatten().astype("int64")
    if ds in ("mnist5", "mnist1"):
        (_, _), (x, y) = keras.datasets.mnist.load_data()
        return np.expand_dims(x.astype("float32")/255.0, -1), y.flatten().astype("int64")
    if ds == "svhn":
        import scipy.io as sio
        d = sio.loadmat(SVHN_MAT)
        x = np.transpose(d["X"], (3,0,1,2)).astype("float32")/255.0
        y = d["y"].flatten().astype("int64"); y[y==10] = 0
        return x, y
    # cifar
    (xtr, _), (x, y) = keras.datasets.cifar10.load_data()
    xtr = xtr.astype("float32")/255.0
    x = x.astype("float32")/255.0
    _CIFAR_MEAN = xtr.mean(axis=0)
    x = x - _CIFAR_MEAN
    return x, y.flatten().astype("int64")

def load_adv_single(cfg, seed_idx, attack):
    """Load adversarial data for ONE attack type (fgsm or bim)."""
    p = f"{cfg['adv_dir']}/{cfg['adv_prefix']}_{attack}_{cfg['adv_key'][seed_idx]}.npy"
    if not os.path.isfile(p):
        return None
    x = np.load(p).astype("float32")
    if cfg['adv_prefix'] == "cifar10":
        x = x - _CIFAR_MEAN
    elif x.max() > 1.5:
        x = x / 255.0
    return x

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

SINGLE_METRICS   = ["DeepGini", "Entropy", "Softmax", "PCS"]
ENSEMBLE_METRICS = ["EnsGini", "Variance", "Disagreement", "MutualInfo", "AdaptiveMI"]
ALL_METRICS      = SINGLE_METRICS + ENSEMBLE_METRICS

def compute_all_scores(P_pool, oracle_idx, setting):
    """
    P_pool: [5, N, C] probs from all models on the pool.
    oracle_idx: which model is the oracle.
    setting: 'exclusive' or 'inclusive'.
    Returns dict metric_name -> [N] scores (higher = prioritise first).
    """
    op = P_pool[oracle_idx]                                   # oracle probs
    if setting == "exclusive":
        ens = P_pool[np.arange(len(P_pool)) != oracle_idx]   # [4, N, C]
    else:
        ens = P_pool                                          # [5, N, C]
    mp = ens.mean(0)                                          # [N, C]

    ent_mean = entropy(mp)
    mi = mutual_info(ens)
    ami = 0.5 * normalize(ent_mean) + 0.5 * normalize(mi)

    return {
        "DeepGini":      deepgini(op),
        "Entropy":       entropy(op),
        "Softmax":       -conf(op),
        "PCS":           -pcs(op),
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

# ============================================================ RUN ONE DATASET
def run_dataset(ds):
    cfg = get_config(ds)
    print(f"\n{'='*70}")
    print(f"DATASET: {cfg['row']}")
    print(f"{'='*70}")

    # check model files
    for p in cfg["models"].values():
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    x_clean, y_clean = load_clean(ds)
    models = {n: keras.models.load_model(p, compile=False)
              for n, p in cfg["models"].items()}
    names = list(models.keys())

    # clean probs [5, N, C]
    P_clean = np.stack([models[n].predict(x_clean, batch_size=256, verbose=0)
                        for n in names], axis=0)
    acc0 = (P_clean[0].argmax(1) == y_clean).mean()
    print(f"  Clean acc (M1) = {acc0:.4f}")

    # results[attack][setting][metric] = list of per-oracle APFDs
    res = {a: {"exclusive": {m: [] for m in ALL_METRICS},
               "inclusive": {m: [] for m in ALL_METRICS}} for a in ATTACKS}

    for attack in ATTACKS:
        print(f"\n  --- {attack.upper()} ---")
        for oi, oname in enumerate(names):
            si = int(oname.replace("seed", ""))
            x_adv = load_adv_single(cfg, si, attack)
            if x_adv is None:
                print(f"    [skip] oracle {oname}: adv file not found")
                continue

            # build pool
            P_adv = np.stack([models[n].predict(x_adv, batch_size=256, verbose=0)
                              for n in names], axis=0)  # [5, N_adv, C]
            if MIX_CLEAN:
                P_pool = np.concatenate([P_adv, P_clean], axis=1)
                y_pool = np.concatenate([y_clean[:x_adv.shape[0]], y_clean])
                # ^ use y_clean[:N_adv] for adv half (aligned), y_clean for clean half
                # Actually: adv is generated from all test images, so len(x_adv)==N_TEST
                y_pool = np.concatenate([y_clean, y_clean])
            else:
                P_pool = P_adv
                y_pool = y_clean

            faults = (P_pool[oi].argmax(1) != y_pool)
            fr = faults.mean()

            for setting in ["exclusive", "inclusive"]:
                scores = compute_all_scores(P_pool, oi, setting)
                for m in ALL_METRICS:
                    a_val = apfd(scores[m], faults)
                    res[attack][setting][m].append(a_val)

            print(f"    Oracle {oname}: pool={len(y_pool)}, fault_rate={fr:.3f}")

    # cleanup
    del models; tf.keras.backend.clear_session(); gc.collect()
    return cfg["row"], res


# ============================================================ MAIN
all_results = {}
for ds in DATASETS:
    try:
        row_name, res = run_dataset(ds)
        all_results[row_name] = res
    except Exception as e:
        print(f"\n  SKIPPED {ds}: {type(e).__name__}: {e}")

# ============================================================ SUMMARY TABLE
print(f"\n\n{'='*90}")
print("PANEL B SUMMARY — Mean APFD over 5 oracle rounds (Exclusive vs Inclusive)")
print(f"{'='*90}")
print(f"{'Dataset':<28s} {'Attack':6s} {'Metric':<14s} {'Exclusive':>10s} {'Inclusive':>10s} {'Gain(pp)':>10s}")
print("-" * 90)

for ds_name, res in all_results.items():
    for attack in ATTACKS:
        for m in ALL_METRICS:
            e_vals = res[attack]["exclusive"][m]
            i_vals = res[attack]["inclusive"][m]
            if not e_vals or not i_vals:
                continue
            e_mean = np.nanmean(e_vals)
            i_mean = np.nanmean(i_vals)
            gain = (i_mean - e_mean) * 100  # percentage points
            print(f"{ds_name:<28s} {attack.upper():6s} {m:<14s} {e_mean:10.4f} {i_mean:10.4f} {gain:>+10.2f}")
        print()

# ============================================================ ENSEMBLE-ONLY COMPACT SUMMARY
print(f"\n{'='*90}")
print("COMPACT — Ensemble metrics only (for paper table)")
print(f"{'='*90}")
print(f"{'Dataset':<28s} {'Attack':6s} {'Metric':<14s} {'Exclusive':>10s} {'Inclusive':>10s} {'Gain(pp)':>10s}")
print("-" * 90)
for ds_name, res in all_results.items():
    for attack in ATTACKS:
        for m in ENSEMBLE_METRICS:
            e_vals = res[attack]["exclusive"][m]
            i_vals = res[attack]["inclusive"][m]
            if not e_vals or not i_vals:
                continue
            e_mean = np.nanmean(e_vals)
            i_mean = np.nanmean(i_vals)
            gain = (i_mean - e_mean) * 100
            print(f"{ds_name:<28s} {attack.upper():6s} {m:<14s} {e_mean:10.4f} {i_mean:10.4f} {gain:>+10.2f}")
        print()

print("\nINTERPRETATION:")
print("  Positive gain = inclusive > exclusive = oracle inclusion helps.")
print("  Expected: BIM gap > FGSM gap (BIM has lower transfer rate).")
print("  Single-model metrics should show ~0 gain (oracle not in ensemble).")
print("\nDone.")