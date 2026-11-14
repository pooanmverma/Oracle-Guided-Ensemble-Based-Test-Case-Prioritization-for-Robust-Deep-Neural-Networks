#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fashion-MNIST adversarial generation for ResNet-20, seeds 0-4.
Generates FGSM (best epsilon) and BIM white-box adversarial examples.

Models:
./models/resnet20_fmnist_seed{seed}.h5

Outputs:
./adv_data_fashion/
    fashion_fgsm_seed{seed}.npy
    fashion_fgsm_seed{seed}_label.npy
    fashion_bim_seed{seed}.npy
    fashion_bim_seed{seed}_label.npy
"""

import os
import gc

# Thread caps BEFORE NumPy import
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import numpy as np
import tensorflow as tf


# =====================================================
# PARAMS
# =====================================================

BATCH = 256
SEEDS = [0, 1, 2, 3,4]

# FGSM: choose, for each test sample, the epsilon giving
# the larger adversarial loss.
FGSM_EPS_LIST = [0.01, 0.1]

# BIM
BIM_EPS = 0.01
BIM_ALPHA = 0.002
BIM_STEPS = 10

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(BASE, "models")
SAVE_DIR = os.path.join(BASE, "adv_data_fashion20")

os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

def load_fashion():
    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_test = x_test.astype("float32") / 255.0
    x_test = x_test[..., np.newaxis]
    y_test = y_test.astype("int64").flatten()

    return x_test, y_test


# =====================================================
# FGSM (BEST EPSILON)
# =====================================================

def fgsm_best(model, x, y, eps_list):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int64)

    best_adv = tf.identity(x)
    best_loss = tf.fill(
        [tf.shape(x)[0]],
        tf.constant(-np.inf, dtype=tf.float32)
    )

    for eps in eps_list:
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = model(x, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred)

        grad = tape.gradient(loss, x)

        adv = x + eps * tf.sign(grad)
        adv = tf.clip_by_value(adv, 0.0, 1.0)

        adv_pred = model(adv, training=False)
        adv_loss = tf.keras.losses.sparse_categorical_crossentropy(y, adv_pred)

        better = adv_loss > best_loss
        mask = tf.reshape(better, (-1, 1, 1, 1))

        best_adv = tf.where(mask, adv, best_adv)
        best_loss = tf.where(better, adv_loss, best_loss)

    return best_adv.numpy()


# =====================================================
# BIM
# =====================================================

def bim_attack(model, x, y, eps, alpha, steps):
    x0 = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int64)
    x_adv = tf.identity(x0)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            pred = model(x_adv, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred)

        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(grad)

        # Project onto the L-infinity epsilon ball around x0.
        x_adv = tf.clip_by_value(x_adv, x0 - eps, x0 + eps)
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)

    return x_adv.numpy()


# =====================================================
# BATCH GENERATION
# =====================================================

def generate_fgsm(model, x_test, y_test):
    adv_list = []

    for i in range(0, len(x_test), BATCH):
        xb = x_test[i:i + BATCH]
        yb = y_test[i:i + BATCH]
        adv_list.append(
            fgsm_best(model, xb, yb, FGSM_EPS_LIST)
        )

    return np.concatenate(adv_list, axis=0).astype("float32")


def generate_bim(model, x_test, y_test):
    adv_list = []

    for i in range(0, len(x_test), BATCH):
        xb = x_test[i:i + BATCH]
        yb = y_test[i:i + BATCH]
        adv_list.append(
            bim_attack(
                model,
                xb,
                yb,
                BIM_EPS,
                BIM_ALPHA,
                BIM_STEPS,
            )
        )

    return np.concatenate(adv_list, axis=0).astype("float32")


# =====================================================
# MAIN
# =====================================================

def main():
    print("\n============================================")
    print(" FASHION-MNIST RESNET-20 ADV GENERATION")
    print(" FGSM + BIM | SEEDS 0-4")
    print("============================================\n")

    x_test, y_test = load_fashion()

    print("Test samples:", len(x_test))
    print("Test shape  :", x_test.shape)
    print("Save dir    :", SAVE_DIR)

    for seed in SEEDS:
        print("\n" + "=" * 60)
        print(f"Generating adversarial examples for SEED = {seed}")
        print("=" * 60)

        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.backend.clear_session()

        model_path = os.path.join(
            MODEL_DIR,
            f"resnet20_fmnist_seed{seed}.h5"
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Missing model file:\n{model_path}"
            )

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        print("Loaded:", model_path)

        # -------------------------------------------------
        # FGSM
        # -------------------------------------------------
        print("Running FGSM...")

        x_fgsm = generate_fgsm(
            model,
            x_test,
            y_test
        )

        fgsm_path = os.path.join(
            SAVE_DIR,
            f"fashion_fgsm_seed{seed}.npy"
        )
        fgsm_label_path = os.path.join(
            SAVE_DIR,
            f"fashion_fgsm_seed{seed}_label.npy"
        )

        np.save(fgsm_path, x_fgsm)
        np.save(fgsm_label_path, y_test)

        print("Saved FGSM:", fgsm_path, x_fgsm.shape)
        print("Saved FGSM labels:", fgsm_label_path, y_test.shape)

        # -------------------------------------------------
        # BIM
        # -------------------------------------------------
        print("Running BIM...")

        x_bim = generate_bim(
            model,
            x_test,
            y_test
        )

        bim_path = os.path.join(
            SAVE_DIR,
            f"fashion_bim_seed{seed}.npy"
        )
        bim_label_path = os.path.join(
            SAVE_DIR,
            f"fashion_bim_seed{seed}_label.npy"
        )

        np.save(bim_path, x_bim)
        np.save(bim_label_path, y_test)

        print("Saved BIM :", bim_path, x_bim.shape)
        print("Saved BIM labels:", bim_label_path, y_test.shape)

        print(f"Seed {seed} completed.")

        del model, x_fgsm, x_bim
        gc.collect()

    print("\n============================================")
    print(" All five seeds finished successfully.")
    print("============================================")


if __name__ == "__main__":
    main()
