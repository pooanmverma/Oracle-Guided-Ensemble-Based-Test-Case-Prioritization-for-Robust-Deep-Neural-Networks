# ==============================================
# SVHN-C Rotating Oracle Framework (FULL METRICS)
# ==============================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np

# ==============================================
# PARAMETERS
# ==============================================

BATCH = 256

MODEL_PATHS = {
    "M0": "./model/svhn_seed0.h5",
    "M1": "./model/svhn_seed1.h5",
    "M2": "./model/svhn_seed2.h5",
    "M3": "./model/svhn_seed3.h5",
    "M4": "./model/svhn_seed4.h5",
}

SVHNC_DIR = "../svhn_c"

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

def load_svhn_c(corruption, severity):
    x = np.load(f"{SVHNC_DIR}/{corruption}_{severity}.npy")
    y = np.load(f"{SVHNC_DIR}/labels.npy")
    x = x.astype("float32") / 255.0
    return x, y

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

# ✅ CORRECT PCS (margin p1 - p2)
def pcs(p):
    sorted_p = np.sort(p, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]

# ✅ CORRECT SOFTMAX (confidence)
def vanilla_softmax(p):
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

    print("\n==============================")
    print(" SVHN-C FULL EVALUATION ")
    print("==============================\n")

    # Load models
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

                x, y = load_svhn_c(corruption, severity)

                # Predictions
                all_probs = []

                all_probs.append(
                    models[oracle].predict(x, batch_size=BATCH, verbose=0)
                )

                for name in model_names:
                    if name != oracle:
                        all_probs.append(
                            models[name].predict(x, batch_size=BATCH, verbose=0)
                        )

                all_probs = np.stack(all_probs, axis=0)

                oracle_probs = all_probs[0]
                ens_probs    = all_probs

                mean_probs = np.mean(ens_probs, axis=0)

                # Faulty samples
                pred = np.argmax(oracle_probs, axis=1)
                faulty = (pred != y).astype(int)

                # ======================================
                # METRICS
                # ======================================

                gini_single = deepgini(oracle_probs)
                gini_ens    = deepgini(mean_probs)

                var_score   = ensemble_variance(ens_probs)
                ent_score   = entropy(oracle_probs)
                dis_score   = disagreement(mean_probs)
                mi_score    = mutual_information(ens_probs)

                hybrid = gini_ens * var_score

                gini_n = normalize(gini_single)
                var_n  = normalize(var_score)

                ent_n = normalize(ent_score)
                mi_n  = normalize(mi_score)
                adaptive_mi = (0.5*ent_n + 0.5*mi_n)

                # ✅ FIXED METRICS
                pcs_score     = pcs(oracle_probs)
                softmax_score = vanilla_softmax(oracle_probs)
                random_scores = random_score(len(oracle_probs))

                # ======================================
                # RANKINGS
                # ======================================

                rankings = {
                    "GINI": np.argsort(-gini_single),
                    "ENSGINI": np.argsort(-gini_ens),
                    "VAR": np.argsort(-var_score),
                    "ENTROPY": np.argsort(-ent_score),
                    "DIS": np.argsort(-dis_score),
                    "MI": np.argsort(-mi_score),
                    "HYBRID": np.argsort(-hybrid),
                    "ADAPTIVE_MI": np.argsort(-adaptive_mi),

                    # ✅ confidence metrics → ascending
                    "PCS": np.argsort(pcs_score),
                    "SOFTMAX": np.argsort(softmax_score),

                    "RANDOM": np.argsort(-random_scores)
                }

                for key in rankings:
                    results[key].append(APFD(rankings[key], faulty))

                print(f"{corruption} | S{severity} done")

    # ==========================================
    # FINAL RESULTS
    # ==========================================

    print("\n==============================")
    print(" FINAL APFD (Mean ± Std)")
    print("==============================")

    for key in results:
        print(f"{key:15s}: {np.mean(results[key]):.4f} ± {np.std(results[key]):.4f}")

    print("\nDone.")

# ==============================================
# RUN
# ==============================================

if __name__ == "__main__":
    main()