# ==============================================
# Fashion-MNIST-C Rotating Oracle Framework
# Exclusive Ensemble | ALL METRICS (FINAL)
# ==============================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
from datasets import load_dataset

# ==============================================
# PARAMETERS
# ==============================================

BATCH = 256

MODEL_PATHS = {
    "M0": "./models/fashion_seed0.h5",
    "M1": "./models/fashion_seed1.h5",
    "M2": "./models/fashion_seed2.h5",
    "M3": "./models/fashion_seed3.h5",
    "M4": "./models/fashion_seed4.h5",
}

DATA_DIR = "fashion-mnist-c"

# ✅ FIXED (aligned with SVHN-C)
CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise",
    "gaussian_blur","motion_blur","zoom_blur",
    "fog","snow","spatter",
    "brightness","contrast","jpeg_compression",
    "translate","rotate","scale"
]

SEVERITIES = [1,2,3,4,5]

# ==============================================
# LOAD DATA
# ==============================================

def load_fmnist_c(corruption, severity):
    X = np.load(os.path.join(DATA_DIR, corruption + ".npy"))
    y = np.load(os.path.join(DATA_DIR, "labels.npy"))

    start = (severity - 1) * 10000
    end   = severity * 10000

    X = X[start:end].astype("float32") / 255
    y = y[start:end]

    X = np.expand_dims(X, -1)

    return X, y

# ==============================================
# METRICS
# ==============================================

def deepgini(p):
    return 1 - np.sum(p**2, axis=1)

def entropy(p):
    return -np.sum(p*np.log(p+1e-12), axis=1)

def ensemble_variance(probs):
    return np.mean(np.var(probs, axis=0), axis=1)

def disagreement(mean_probs):
    return 1 - np.max(mean_probs, axis=1)

def mutual_information(ens):
    mean_probs = np.mean(ens, axis=0)
    H_mean = -np.sum(mean_probs*np.log(mean_probs+1e-12), axis=1)
    H_models = -np.sum(ens*np.log(ens+1e-12), axis=2)
    return H_mean - np.mean(H_models, axis=0)

def normalize(x):
    return (x-x.min())/(x.max()-x.min()+1e-12)

# ✅ CORRECT PCS (margin)
def pcs(p):
    sorted_p = np.sort(p, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]

# ✅ SOFTMAX confidence
def softmax_conf(p):
    return np.max(p, axis=1)

def random_score(n):
    return np.random.rand(n)

# ==============================================
# APFD
# ==============================================

def APFD(rank, faulty):
    n = len(faulty)
    m = faulty.sum()

    if m == 0:
        return 0.0

    s = 0
    for i, idx in enumerate(rank):
        if faulty[idx]:
            s += i + 1

    return 1 - s/(n*m) + 1/(2*n)

# ==============================================
# MAIN
# ==============================================

def main():

    print("\n==== Fashion-MNIST-C ROTATING ORACLE ====\n")

    models = {}
    for name, path in MODEL_PATHS.items():
        models[name] = tf.keras.models.load_model(path, compile=False)
        print("Loaded:", name)

    model_names = list(models.keys())

    results = {
        "GINI":[],
        "ENSGINI":[],
        "VAR":[],
        "ENTROPY":[],
        "DIS":[],
        "MI":[],
        "HYBRID":[],
        "ADAPTIVE_MI":[],
        "PCS":[],
        "SOFTMAX":[],
        "RANDOM":[]
    }

    # ==========================================
    # ROTATING ORACLE
    # ==========================================

    for oracle in model_names:

        print("\nORACLE:", oracle)

        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:

                X, y = load_fmnist_c(corruption, severity)

                # predictions
                all_probs = []

                all_probs.append(
                    models[oracle].predict(X, batch_size=BATCH, verbose=0)
                )

                for name in model_names:
                    if name != oracle:
                        all_probs.append(
                            models[name].predict(X, batch_size=BATCH, verbose=0)
                        )

                all_probs = np.stack(all_probs, axis=0)

                oracle_probs = all_probs[0]
                ensemble_probs = all_probs[1:]
                mean_probs = np.mean(ensemble_probs, axis=0)

                # faulty
                pred = np.argmax(oracle_probs, axis=1)
                faulty = (pred != y).astype(int)

                # metrics
                gini_single = deepgini(oracle_probs)
                gini_ens = deepgini(mean_probs)
                var_score = ensemble_variance(ensemble_probs)
                ent_score = entropy(oracle_probs)
                dis_score = disagreement(mean_probs)
                mi_score = mutual_information(ensemble_probs)

                hybrid = gini_ens * var_score

                gini_n = normalize(gini_single)
                var_n = normalize(var_score)

                ent_n = normalize(ent_score)
                mi_n = normalize(mi_score)
                adaptive_mi = 0.5*ent_n + 0.5*mi_n

                # ✅ new metrics
                pcs_score = pcs(oracle_probs)
                softmax_score = softmax_conf(oracle_probs)
                rand_score = random_score(len(pcs_score))

                # rankings
                rankings = {
                    "GINI": np.argsort(-gini_single),
                    "ENSGINI": np.argsort(-gini_ens),
                    "VAR": np.argsort(-var_score),
                    "ENTROPY": np.argsort(-ent_score),
                    "DIS": np.argsort(-dis_score),
                    "MI": np.argsort(-mi_score),
                    "HYBRID": np.argsort(-hybrid),
                    "ADAPTIVE_MI": np.argsort(-adaptive_mi),

                    # confidence → ascending
                    "PCS": np.argsort(pcs_score),
                    "SOFTMAX": np.argsort(softmax_score),

                    "RANDOM": np.argsort(-rand_score)
                }

                for key in rankings:
                    results[key].append(APFD(rankings[key], faulty))

                print(corruption, severity, "done")

    # ==========================================
    # FINAL RESULTS
    # ==========================================

    print("\n===== FINAL APFD =====")

    for key in results:
        print(
            f"{key:15s}: "
            f"{np.mean(results[key]):.4f} ± "
            f"{np.std(results[key]):.4f}"
        )

if __name__ == "__main__":
    main()