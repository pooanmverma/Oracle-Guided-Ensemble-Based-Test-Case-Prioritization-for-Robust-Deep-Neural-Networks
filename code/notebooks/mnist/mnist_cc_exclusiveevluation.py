# ==============================================
# MNIST-C Rotating Oracle Framework (FINAL)
# Exclusive Ensemble | ALL METRICS (FIXED)
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
    "M0": "./models/model_mnist_seed0.h5",
    "M1": "./models/model_mnist_seed1.h5",
    "M2": "./models/model_mnist_seed2.h5",
    "M3": "./models/model_mnist_seed3.h5",
    "M4": "./models/model_mnist_seed4.h5",
}

MNISTC_DIR = "./mnist_c"

CORRUPTIONS = [
    "brightness","canny_edges","dotted_line","fog",
    "glass_blur","impulse_noise","motion_blur",
    "rotate","scale","shear","shot_noise",
    "spatter","stripe","translate","zigzag"
]

SEVERITIES = [1,2,3,4,5]

# ==============================================
# LOAD DATA
# ==============================================

def load_mnist_c(corruption, severity):
    x = np.load(os.path.join(MNISTC_DIR, corruption, "test_images.npy"))
    y = np.load(os.path.join(MNISTC_DIR, corruption, "test_labels.npy"))

    x = x.astype("float32") / 255
    x = np.expand_dims(x, -1)

    N = len(x)
    chunk = N // 5

    start = (severity - 1) * chunk
    end   = severity * chunk

    return x[start:end], y[start:end]

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

# ==============================================
# ADDITIONAL METRICS (FIXED)
# ==============================================

def pcs(p):
    sorted_p = np.sort(p, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]

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

    print("\n==== MNIST-C ROTATING ORACLE (EXCLUSIVE) ====\n")

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
        "ADAPTIVE_VAR":[],
        "ADAPTIVE_MI":[],
        "PCS":[],
        "SOFTMAX":[],
        "RANDOM":[]
    }

    # ==========================================
    # ROTATING ORACLE (EXCLUSIVE)
    # ==========================================

    for oracle in model_names:

        print("\nORACLE:", oracle)

        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:

                x_all, y_all = load_mnist_c(corruption, severity)

                if len(x_all) == 0:
                    print("SKIP", corruption, severity)
                    continue

                # ---------- Prediction ----------
                all_probs = []

                # oracle first
                all_probs.append(
                    models[oracle].predict(x_all, batch_size=BATCH, verbose=0)
                )

                # ensemble WITHOUT oracle
                for name in model_names:
                    if name != oracle:
                        all_probs.append(
                            models[name].predict(x_all, batch_size=BATCH, verbose=0)
                        )

                all_probs = np.stack(all_probs, axis=0)

                oracle_probs = all_probs[0]
                ensemble_probs = all_probs[1:]

                mean_probs = np.mean(ensemble_probs, axis=0)

                # ---------- Faulty ----------
                pred = np.argmax(oracle_probs, axis=1)
                faulty = (pred != y_all).astype(int)

                # ---------- Metrics ----------
                gini_single = deepgini(oracle_probs)
                gini_ens = deepgini(mean_probs)
                var_score = ensemble_variance(ensemble_probs)
                ent_score = entropy(oracle_probs)
                dis_score = disagreement(mean_probs)
                mi_score = mutual_information(ensemble_probs)

                hybrid = gini_ens * var_score

                gini_n = normalize(gini_single)
                var_n = normalize(var_score)
                adaptive_var = (1-var_n)*gini_n + var_n*var_n

                ent_n = normalize(ent_score)
                mi_n = normalize(mi_score)
                adaptive_mi = 0.5*ent_n + 0.5*mi_n

                pcs_score = pcs(oracle_probs)
                softmax_score = softmax_conf(oracle_probs)
                rand_score = random_score(len(pcs_score))

                # ---------- Rankings ----------
                rankings = {
                    "GINI": np.argsort(-gini_single),
                    "ENSGINI": np.argsort(-gini_ens),
                    "VAR": np.argsort(-var_score),
                    "ENTROPY": np.argsort(-ent_score),
                    "DIS": np.argsort(-dis_score),
                    "MI": np.argsort(-mi_score),
                    "HYBRID": np.argsort(-hybrid),
                    "ADAPTIVE_VAR": np.argsort(-adaptive_var),
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
        print(f"{key:15s}: {np.mean(results[key]):.4f} ± {np.std(results[key]):.4f}")


if __name__ == "__main__":
    main()