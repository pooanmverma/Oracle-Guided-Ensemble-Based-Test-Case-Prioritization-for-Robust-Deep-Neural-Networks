#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=========================================================
SVHN GUIDANCE PIPELINE  (RQ3, DeepGini-style)

Single file, three stages:
    STAGE = "split" | "generate" | "retrain" | "all"

SVHN-specific handling (differs from MNIST/Fashion):
  * input is 32x32x3 (RGB)
  * adversarial images live in adv_data_svhn/ named svhn_{attack}_{i}.npy,
    already scaled to [0,1], in TEST-SET ORDER (26,032 images)
  * labels come from svhn/test_32x32.mat -> y, with the 10->0 remap;
    the same test labels apply to every seed's adv (only the attack target
    model differs, not the underlying images)
  * models are model/svhn_seed{i}.h5   (folder 'model', not 'models')
  * normalization is /255 only (NO mean subtraction), matching training

Sanity confirmed: clean acc 0.855, adv (FGSM) acc 0.03 -> attack is strong,
so retraining has large headroom, exactly the DeepGini RQ3 regime.

Resume-safe: retrain appends each result to CSV immediately.
=========================================================
"""

import os
import random
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers, models as kmodels
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

# =====================================================
# WHAT TO RUN
# =====================================================

STAGE = "all"          # "split" | "generate" | "retrain" | "all"

# =====================================================
# PARAMETERS
# =====================================================

BASE = os.path.dirname(os.path.abspath(__file__))
ADV_DIR = f"{BASE}/adv_data_svhn"
MAT_TEST = f"{BASE}/svhn/test_32x32.mat"
OUTPUT_DIR = f"{BASE}/guidance_data_svhn"
MODEL_DIR = f"{BASE}/model"                     # folder is 'model' (singular)
RETRAIN_OUT = f"{BASE}/retrained_models_svhn"

# seed key -> (model file, adv index)
SEEDS = ["seed0", "seed1", "seed2", "seed3", "seed4"]
ADV_IDX = {f"seed{i}": i for i in range(5)}     # seed0 -> svhn_fgsm_0.npy

TOP_PERCENTAGES = list(range(1, 11))
EPOCHS = 15
BATCH_SIZE = 64
BATCH = 256
NUM_CLASSES = 10
RANDOM_SEED = 42

METRIC_NAMES = ["DeepGini", "Entropy", "EnsGini", "Variance", "Disagreement",
 "MutualInfo", "AdaptiveMI", "Softmax", "PCS"]

# =====================================================
# HELPERS
# =====================================================

def set_seed(seed):
    s = int(seed.replace("seed", ""))
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

def _to_int(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == NUM_CLASSES:
        return y.argmax(axis=1).astype("int64")
    return y.reshape(-1).astype("int64")

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

def load_test_labels():
    y = sio.loadmat(MAT_TEST)["y"].reshape(-1)
    y[y == 10] = 0
    return y.astype("int64")

def model_path(seed):
    return os.path.join(MODEL_DIR, f"svhn_{seed}.h5")

# =====================================================
# STAGE 1 : SPLIT  T / V
# =====================================================

def stage_split():
    print("\n" + "=" * 60 + "\nSTAGE 1: SPLIT T / V\n" + "=" * 60)
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y_test = load_test_labels()          # shared test-order labels

    for seed in SEEDS:
        save_dir = os.path.join(OUTPUT_DIR, seed)
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(os.path.join(save_dir, "V_labels.npy")):
            print(f"  {seed}: already split, skipping")
            continue

        i = ADV_IDX[seed]
        fgsm = np.load(f"{ADV_DIR}/svhn_fgsm_{i}.npy").astype("float32")
        bim = np.load(f"{ADV_DIR}/svhn_bim_{i}.npy").astype("float32")

        # each adv block is in test order -> label block == y_test
        images = np.concatenate([fgsm, bim], axis=0)
        labels = np.concatenate([y_test, y_test], axis=0)

        idx = np.random.permutation(len(images))
        images, labels = images[idx], labels[idx]

        split = len(images) // 2
        np.save(os.path.join(save_dir, "T_images.npy"), images[:split])
        np.save(os.path.join(save_dir, "T_labels.npy"), labels[:split])
        np.save(os.path.join(save_dir, "V_images.npy"), images[split:])
        np.save(os.path.join(save_dir, "V_labels.npy"), labels[split:])
        print(f"  {seed}: total={len(images)}  T={split}  V={len(images)-split}")

# =====================================================
# METRICS
# =====================================================

def deepgini(p): return 1.0 - np.sum(p ** 2, axis=1)
def entropy(p): return -np.sum(p * np.log(p + 1e-12), axis=1)
def ensemble_variance(probs): return np.mean(np.var(probs, axis=0), axis=1)
def disagreement(mp): return 1.0 - np.max(mp, axis=1)

def mutual_information(ens):
    mp = np.mean(ens, axis=0)
    em = -np.sum(mp * np.log(mp + 1e-12), axis=1)
    emod = -np.sum(ens * np.log(ens + 1e-12), axis=2)
    return em - np.mean(emod, axis=0)

def softmax_confidence(p): return np.max(p, axis=1)
def pcs(p):
    s = -np.sort(-p, axis=1); return s[:, 0] - s[:, 1]

# =====================================================
# STAGE 2 : GENERATE  ranked top-k
# =====================================================

def stage_generate():
    print("\n" + "=" * 60 + "\nSTAGE 2: GENERATE RANKED TOP-K\n" + "=" * 60)

    models = {s: load_model(model_path(s), compile=False) for s in SEEDS}
    names = list(models.keys())

    for oracle in names:
        print(f"\nORACLE: {oracle}")
        odir = os.path.join(OUTPUT_DIR, oracle)
        if os.path.exists(os.path.join(odir, "pcs", "top10_images.npy")):
            print(f"  {oracle}: already generated, skipping")
            continue

        x_T = np.load(os.path.join(odir, "T_images.npy")).astype("float32")
        y_T = _to_int(np.load(os.path.join(odir, "T_labels.npy")))

        all_probs = np.stack(
            [models[n].predict(x_T, batch_size=BATCH, verbose=0) for n in names],
            axis=0)
        oi = names.index(oracle)
        op = all_probs[oi]
        mp = np.mean(all_probs, axis=0)

        gini = deepgini(op); ent = entropy(op)
        gini_ens = deepgini(mp); var = ensemble_variance(all_probs)
        dis = disagreement(mp); mi = mutual_information(all_probs)
        soft = softmax_confidence(op); pcs_s = pcs(op)
        av = (1 - normalize(var)) * normalize(gini) + normalize(var) ** 2
        ami = 0.5 * normalize(ent) + 0.5 * normalize(mi)

        rankings = {
            "DeepGini": np.argsort(-gini), "Entropy": np.argsort(-ent),
            "EnsGini": np.argsort(-gini_ens), "Variance": np.argsort(-var),
            "Disagreement": np.argsort(-dis), "MutualInfo": np.argsort(-mi),
: np.argsort(-av), "AdaptiveMI": np.argsort(-ami),
            "Softmax": np.argsort(soft), "PCS": np.argsort(pcs_s),
        }

        for metric in METRIC_NAMES:
            mdir = os.path.join(odir, metric.lower())
            os.makedirs(mdir, exist_ok=True)
            np.save(os.path.join(mdir, "ranking.npy"), rankings[metric])
            for p in TOP_PERCENTAGES:
                k = max(1, int(len(rankings[metric]) * p / 100))
                idx = rankings[metric][:k]
                np.save(os.path.join(mdir, f"top{p}_images.npy"), x_T[idx])
                np.save(os.path.join(mdir, f"top{p}_labels.npy"), y_T[idx])
            print(f"  {metric:12s} done")

# =====================================================
# MODEL  (LeNet-5, SVHN: 32x32x3)  -- matches training script
# =====================================================

def build_model():
    return kmodels.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(6, (5, 5), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, (5, 5), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(120, activation="relu"),
        layers.Dense(84, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

def load_svhn_train():
    train = sio.loadmat(f"{BASE}/svhn/train_32x32.mat")
    x = np.transpose(train["X"], (3, 0, 1, 2)).astype("float32") / 255.0
    y = train["y"].reshape(-1); y[y == 10] = 0
    return x, y.astype("int64")

# =====================================================
# STAGE 3 : RETRAIN  (resumable)
# =====================================================

def stage_retrain():
    print("\n" + "=" * 60 + "\nSTAGE 3: RETRAIN (resumable)\n" + "=" * 60)
    os.makedirs(RETRAIN_OUT, exist_ok=True)
    csv_path = os.path.join(RETRAIN_OUT, "guidance_results_svhn.csv")

    done = set()
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        done = set(zip(prev.Seed, prev.Metric, prev.TopPercent))
        print(f"  resuming: {len(done)} combos already done")

    print("\n  INITIAL ACC ON V (no retrain):")
    for seed in SEEDS:
        vp = os.path.join(OUTPUT_DIR, seed, "V_images.npy")
        mp = model_path(seed)
        if os.path.exists(vp) and os.path.exists(mp):
            V_img = np.load(vp).astype("float32")
            V_lab = _to_int(np.load(os.path.join(OUTPUT_DIR, seed, "V_labels.npy")))
            m = load_model(mp, compile=False)
            m.compile(loss="sparse_categorical_crossentropy",
                      optimizer="sgd", metrics=["accuracy"])
            _, acc = m.evaluate(V_img, V_lab, verbose=0)
            print(f"    {seed}: {acc:.4f}")

    x_train_orig, y_train_orig = load_svhn_train()

    for seed in SEEDS:
        print(f"\n  === SEED {seed} ===")
        V_img = np.load(os.path.join(OUTPUT_DIR, seed, "V_images.npy")).astype("float32")
        V_lab = _to_int(np.load(os.path.join(OUTPUT_DIR, seed, "V_labels.npy")))

        for metric in [m.lower() for m in METRIC_NAMES]:
            for percent in TOP_PERCENTAGES:
                if (seed, metric, percent) in done:
                    continue
                set_seed(seed); K.clear_session()

                mdir = os.path.join(OUTPUT_DIR, seed, metric)
                X_top = np.load(os.path.join(mdir, f"top{percent}_images.npy")).astype("float32")
                Y_top = _to_int(np.load(os.path.join(mdir, f"top{percent}_labels.npy")))

                X = np.concatenate([x_train_orig, X_top], axis=0)
                Y = np.concatenate([y_train_orig, Y_top], axis=0)
                idx = np.random.permutation(len(X))
                X, Y = X[idx], Y[idx]

                model = build_model()
                model.compile(
                    optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                model.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPOCHS,
                          validation_data=(V_img, V_lab), verbose=0)
                _, acc = model.evaluate(V_img, V_lab, verbose=0)

                row = pd.DataFrame([{
                    "Seed": seed, "Metric": metric,
                    "TopPercent": percent, "Accuracy": acc}])
                row.to_csv(csv_path, mode="a",
                           header=not os.path.exists(csv_path), index=False)
                print(f"    {metric:12s} top{percent:2d}%  acc={acc:.4f}")

    df = pd.read_csv(csv_path)
    pivot = df.groupby(["Metric", "TopPercent"])["Accuracy"].mean().unstack()
    print("\n  Mean accuracy (rows=metric, cols=top%):")
    print(pivot.round(4))
    print("\n  Saved ->", csv_path)

# =====================================================
# MAIN
# =====================================================

def main():
    if STAGE in ("split", "all"):
        stage_split()
    if STAGE in ("generate", "all"):
        stage_generate()
    if STAGE in ("retrain", "all"):
        stage_retrain()
    print("\nDONE.")

if __name__ == "__main__":
    main()