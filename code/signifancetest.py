"""
FULL Table 3 significance in one run — all datasets, all settings.

For every Table 3 row (dataset x model) and every setting
{clean,adversarial} x {inclusive,exclusive}, tests whether the BEST
ensemble metric significantly beats the BEST single-model metric via a
paired bootstrap over test instances. Pools ALL raw p-values across the
whole table and applies ONE Holm-Bonferroni correction.

Just run it:  python significance_table3_ALL.py
(Set SVHN_MAT below; a dataset whose files are missing is skipped, not fatal.)

SAFETY: models loaded READ-ONLY (compile=False). Never call .save().
"""

import os, gc
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ============================================================ CONFIG
BASE  = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
B     = 10000
SEED  = 42
SVHN_MAT = f"{BASE}/data/test_32x32.mat"     # <-- set to your SVHN test .mat
_CIFAR_MEAN = None                           # per-pixel CIFAR train mean, set in load_clean

DATASETS = ["fashion", "fashion20", "cifar", "svhn", "mnist5", "mnist1"]

SINGLE_METRICS   = ["DeepGini", "Entropy", "Softmax", "PCS"]
ENSEMBLE_METRICS = ["EnsGini", "Variance", "Disagreement", "MutualInfo", "AdaptiveMI"]
SETTINGS = [("clean","inclusive"), ("adversarial","inclusive"),
            ("clean","exclusive"), ("adversarial","exclusive")]

# ============================================================ PER-DATASET CONFIG
def get_config(ds):
    if ds == "fashion":
        return dict(models={f"seed{s}": f"{BASE}/models/fashion_seed{s}.h5" for s in SEEDS},
                    adv_dir=f"{BASE}/adv_data_fashion", adv_prefix="fashion",
                    adv_key={s: f"seed{s}" for s in SEEDS}, row="Fashion-MNIST / LeNet-5")
    if ds == "fashion20":
        return dict(models={f"seed{s}": f"{BASE}/models/resnet20_fmnist_seed{s}.h5" for s in SEEDS},
                    adv_dir=f"{BASE}/adv_data_fashion20", adv_prefix="fashion",
                    adv_key={s: f"seed{s}" for s in SEEDS}, row="Fashion-MNIST / ResNet-20")
    if ds == "cifar":
        C = {0:"resnet20_best",1:"resnet20_seed1",2:"resnet20_seed2",3:"resnet20_seed9",4:"resnet20_seed23"}
        return dict(models={f"seed{s}": f"{BASE}/models/{C[s]}.h5" for s in SEEDS},
                    adv_dir=f"{BASE}/adv_data", adv_prefix="cifar10",
                    adv_key={s: C[s] for s in SEEDS}, row="CIFAR-10 / ResNet-20")
    if ds == "svhn":
        return dict(models={f"seed{s}": f"{BASE}/model/svhn_seed{s}.h5" for s in SEEDS},
                    adv_dir=f"{BASE}/adv_data_svhn", adv_prefix="svhn",
                    adv_key={s: str(s) for s in SEEDS}, row="SVHN / LeNet-5")
    if ds == "mnist5":
        return dict(models={f"seed{s}": f"{BASE}/models/lenet1_mnist_adapted_seed{s}.h5" for s in SEEDS},
                    adv_dir=f"{BASE}/adv_data", adv_prefix="mnist_lenet1",
                    adv_key={s: f"seed{s}" for s in SEEDS}, row="MNIST / LeNet-5")
    if ds == "mnist1":
        return dict(models={f"seed{s}": f"{BASE}/models/model_mnist_seed{s}.h5" for s in SEEDS},
                    adv_dir=f"{BASE}/adv_data", adv_prefix="mnist",
                    adv_key={s: f"seed{s}" for s in SEEDS}, row="MNIST / LeNet-1")
    raise ValueError(ds)

def load_clean(ds):
    if ds in ("fashion", "fashion20"):
        (_, _), (x, y) = tf.keras.datasets.fashion_mnist.load_data()
        return np.expand_dims(x.astype("float32")/255.0, -1), y.flatten().astype("int64")
    if ds in ("mnist5", "mnist1"):
        (_, _), (x, y) = tf.keras.datasets.mnist.load_data()
        return np.expand_dims(x.astype("float32")/255.0, -1), y.flatten().astype("int64")
    if ds == "svhn":
        import scipy.io as sio
        d = sio.loadmat(SVHN_MAT)
        x = np.transpose(d["X"], (3,0,1,2)).astype("float32")/255.0
        y = d["y"].flatten().astype("int64"); y[y==10] = 0
        return x, y
    # cifar: per-pixel train-mean subtraction, exactly as the canonical pipeline
    (xtr, _), (x, y) = tf.keras.datasets.cifar10.load_data()
    xtr = xtr.astype("float32")/255.0
    x = x.astype("float32")/255.0
    global _CIFAR_MEAN
    _CIFAR_MEAN = xtr.mean(axis=0)          # (32,32,3)
    x = x - _CIFAR_MEAN
    return x, y.flatten().astype("int64")

def load_adv(cfg, seed_name, x_clean):
    s = int(seed_name.replace("seed", ""))
    xs = []
    for atk in ["fgsm", "bim"]:
        p = f"{cfg['adv_dir']}/{cfg['adv_prefix']}_{atk}_{cfg['adv_key'][s]}.npy"
        xs.append(np.load(p).astype("float32"))
    x = np.concatenate(xs, 0)
    if cfg['adv_prefix'] == "cifar10":
        x = x - _CIFAR_MEAN                 # adv stored in [0,1]; subtract same mean (no /255)
    elif x.max() > 1.5:
        x = x/255.0
    return x

# ============================================================ METRICS / APFD
EPS = 1e-12
def deepgini(p):   return 1 - np.sum(p**2, axis=1)
def entropy(p):    return -np.sum(p*np.log(p+EPS), axis=1)
def ens_var(pr):   return np.mean(np.var(pr, axis=0), axis=1)
def disagree(mp):  return 1 - np.max(mp, axis=1)
def conf(p):       return np.max(p, axis=1)
def pcs(p):
    sp = -np.sort(-p, axis=1); return sp[:,0]-sp[:,1]
def mutual_info(ens):
    mp = ens.mean(0); hm = -np.sum(mp*np.log(mp+EPS),1)
    he = -np.sum(ens*np.log(ens+EPS),2); return hm - he.mean(0)
def normalize(x): return (x-x.min())/(x.max()-x.min()+EPS)

def priority_scores(all_probs, oi, setting):
    op = all_probs[oi]
    ens = all_probs[np.arange(len(all_probs))!=oi] if setting=="exclusive" else all_probs
    mp = ens.mean(0)
    gini=deepgini(op); ent=entropy(op); var=ens_var(ens); mi=mutual_info(ens)
    ent_mean = entropy(mp)                      # AdaptiveMI uses ensemble-mean entropy, not oracle entropy
    ami = 0.5*normalize(ent_mean) + 0.5*normalize(mi)
    return {"DeepGini":gini,"Entropy":ent,"Softmax":-conf(op),"PCS":-pcs(op),
            "EnsGini":deepgini(mp),"Variance":var,"Disagreement":disagree(mp),
            "MutualInfo":mi,"AdaptiveMI":ami}

def apfd_scores(score, faulty):
    n=len(score); m=int(faulty.sum())
    if m==0: return np.nan
    order=np.argsort(-score, kind="stable")
    pos=np.empty(n,dtype=np.int64); pos[order]=np.arange(1,n+1)
    return 1.0 - pos[faulty.astype(bool)].sum()/(n*m) + 1.0/(2*n)

def ci(a): return np.nanpercentile(a,2.5), np.nanpercentile(a,97.5)

# ============================================================ RUN ONE DATASET
def run_dataset(ds, rng):
    cfg = get_config(ds)
    for p in cfg["models"].values():
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    x_clean, y_clean = load_clean(ds)
    models = {n: tf.keras.models.load_model(p, compile=False) for n,p in cfg["models"].items()}
    names = list(models.keys())
    # sanity: clean accuracy of first model
    acc0 = (models[names[0]].predict(x_clean, batch_size=256, verbose=0).argmax(1)==y_clean).mean()
    print(f"  [{ds}] clean acc (M1) = {acc0:.3f}" + ("   <-- LOW! check input preprocessing" if acc0 < 0.5 else ""))

    out = []
    for data_setting, ens_setting in SETTINGS:
        sc = {}; ft = {}
        for oracle in names:
            if data_setting == "clean":
                x_all, y_all = x_clean, y_clean
            else:
                xadv = load_adv(cfg, oracle, x_clean); k = len(xadv)//len(x_clean)
                x_all = np.concatenate([x_clean, xadv],0); y_all = np.tile(y_clean,1+k)
            ap = np.stack([models[n].predict(x_all,batch_size=256,verbose=0) for n in names],0)
            oi = names.index(oracle)
            ft[oracle] = (ap[oi].argmax(1)!=y_all).astype(int)
            sc[oracle] = priority_scores(ap, oi, ens_setting)
        N = len(ft[names[0]])
        point = {m: np.mean([apfd_scores(sc[o][m], ft[o]) for o in names])
                 for m in SINGLE_METRICS+ENSEMBLE_METRICS}
        best_s = max(SINGLE_METRICS, key=lambda m: point[m])
        best_e = max(ENSEMBLE_METRICS, key=lambda m: point[m])
        diffs = np.empty(B)
        for b in range(B):
            idx = rng.integers(0,N,N)
            ae = np.nanmean([apfd_scores(sc[o][best_e][idx], ft[o][idx]) for o in names])
            asg= np.nanmean([apfd_scores(sc[o][best_s][idx], ft[o][idx]) for o in names])
            diffs[b] = ae - asg
        lo,hi = ci(diffs)
        p_raw = min(1.0, 2.0*min((diffs<=0).mean(), (diffs>=0).mean()))
        out.append(dict(row=cfg["row"], ds=ds, setting=f"{data_setting}+{ens_setting}",
                        best_s=best_s, aps=point[best_s], best_e=best_e, ape=point[best_e],
                        d=point[best_e]-point[best_s], lo=lo, hi=hi, p_raw=p_raw))
        print(f"    {data_setting:11s} {ens_setting:9s}  "
              f"{best_s}={point[best_s]:.4f}  {best_e}={point[best_e]:.4f}  d={point[best_e]-point[best_s]:+.4f}")
    del models; tf.keras.backend.clear_session(); gc.collect()
    return out

# ============================================================ MAIN
def holm(pv):
    m=len(pv); order=np.argsort(pv); adj=np.empty(m); run=0.0
    for r,i in enumerate(order):
        run=max(run,(m-r)*pv[i]); adj[i]=min(run,1.0)
    return adj

all_cells = []
for ds in DATASETS:
    print(f"\n=== {ds} ===")
    try:
        rng = np.random.default_rng(SEED)   # reset per dataset for reproducibility
        all_cells += run_dataset(ds, rng)
    except Exception as e:
        print(f"  SKIPPED {ds}: {type(e).__name__}: {e}")

if not all_cells:
    print("\nNo datasets ran. Check paths."); raise SystemExit

# ONE Holm across the whole table
p_adj = holm(np.array([c["p_raw"] for c in all_cells]))

print(f"\n{'='*104}")
print(f"FULL TABLE 3 SIGNIFICANCE  ({len(all_cells)} cells, ONE Holm correction across all)")
print(f"{'='*104}")
print(f"{'Row':26s} {'Setting':22s} {'Single':>8s} {'Ens':>8s} {'dAPFD':>8s} {'95% CI':>20s} {'Holm p':>9s}")
print("-"*104)
for c, pa in zip(all_cells, p_adj):
    padj = f"<{1.0/B:.0e}" if pa==0 else f"{pa:.4f}"
    sig = " *" if ((c['lo']>0 or c['hi']<0) and pa<0.05) else ""
    print(f"{c['row']:26s} {c['setting']:22s} {c['aps']:8.4f} {c['ape']:8.4f} {c['d']:+8.4f} "
          f"[{c['lo']:+.4f},{c['hi']:+.4f}] {padj:>9s}{sig}")

nsig = sum(1 for c,pa in zip(all_cells,p_adj) if (c['lo']>0 or c['hi']<0) and pa<0.05)
npos = sum(1 for c in all_cells if c['d']>0)
print("-"*104)
print(f"  Ensemble higher in {npos}/{len(all_cells)} cells; significant (Holm) in {nsig}/{len(all_cells)}.")
print("  * = 95% CI excludes 0 AND Holm-adjusted p < 0.05")
print("\nCaveats for the paper:")
print("  - 'Best' metric picked on the same data (mild optimistic bias).")
print("  - Bootstrap resamples TEST INSTANCES; inference is conditional on these trained models.")
print("\nDone.")