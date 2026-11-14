#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# MNIST ADVERSARIAL GENERATION
# FGSM (Best-of-Eps) + BIM (Iter-FGSM)
# No Foolbox (Pure TF)
# Generate adversarial examples separately
# for each independently trained model
#
# ALSO SAVE THE CORRESPONDING MNIST TEST LABELS
# so the guidance/retraining pipeline can use them.
# =====================================

import tensorflow as tf
import numpy as np
import os

BATCH = 256
FGSM_EPS_LIST = [0.01, 0.1]
BIM_EPS = 0.01
BIM_ALPHA = 0.002
BIM_STEPS = 10
SAVE_DIR = "adv_data"

MODEL_PATHS = [
    "models/model_mnist_seed0.h5",
    "models/model_mnist_seed1.h5",
    "models/model_mnist_seed2.h5",
    "models/model_mnist_seed3.h5",
    "models/model_mnist_seed4.h5",
]

os.makedirs(SAVE_DIR, exist_ok=True)


def fgsm_attack_best(model, x, y, eps_list):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    best_adv = tf.identity(x)
    best_loss = tf.zeros(len(x))

    for eps in eps_list:
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = model(x, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred)

        grad = tape.gradient(loss, x)
        adv = x + eps * tf.sign(grad)
        adv = tf.clip_by_value(adv, 0, 1)

        adv_pred = model(adv, training=False)
        adv_loss = tf.keras.losses.sparse_categorical_crossentropy(y, adv_pred)
        better = adv_loss > best_loss
        mask = tf.reshape(better, (-1, 1, 1, 1))
        best_adv = tf.where(mask, adv, best_adv)
        best_loss = tf.where(better, adv_loss, best_loss)

    return best_adv.numpy()


def bim_attack(model, x, y, eps, alpha, steps):
    x0 = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_adv = tf.identity(x0)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            pred = model(x_adv, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred)

        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, x0 - eps, x0 + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)

    return x_adv.numpy()


def load_mnist():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test = y_test.flatten()
    return x_test, y_test


def save_labels(y_test):
    # FGSM and BIM both use the original MNIST test-set ordering.
    np.save(f"{SAVE_DIR}/mnist_fgsm_label.npy", y_test)
    np.save(f"{SAVE_DIR}/mnist_bim_label.npy", y_test)
    print(f"FGSM labels saved: {SAVE_DIR}/mnist_fgsm_label.npy {y_test.shape}")
    print(f"BIM labels saved: {SAVE_DIR}/mnist_bim_label.npy {y_test.shape}")


def generate_adv_for_model(model, x_test, y_test, seed):
    print(f"\nRunning FGSM for seed{seed}...")
    adv_fgsm = []

    for i in range(0, len(x_test), BATCH):
        x_batch = x_test[i:i + BATCH]
        y_batch = y_test[i:i + BATCH]
        adv_fgsm.append(fgsm_attack_best(model, x_batch, y_batch, FGSM_EPS_LIST))

    adv_fgsm = np.concatenate(adv_fgsm, axis=0)
    fgsm_path = f"{SAVE_DIR}/mnist_fgsm_seed{seed}.npy"
    np.save(fgsm_path, adv_fgsm)
    print(f"FGSM saved: {fgsm_path} {adv_fgsm.shape}")

    print(f"\nRunning BIM for seed{seed}...")
    adv_bim = []

    for i in range(0, len(x_test), BATCH):
        x_batch = x_test[i:i + BATCH]
        y_batch = y_test[i:i + BATCH]
        adv_bim.append(bim_attack(model, x_batch, y_batch, BIM_EPS, BIM_ALPHA, BIM_STEPS))

    adv_bim = np.concatenate(adv_bim, axis=0)
    bim_path = f"{SAVE_DIR}/mnist_bim_seed{seed}.npy"
    np.save(bim_path, adv_bim)
    print(f"BIM saved: {bim_path} {adv_bim.shape}")


def generate_adv():
    print("\n====================================")
    print(" MNIST ADV GENERATION (ALL MODELS) ")
    print("====================================\n")

    x_test, y_test = load_mnist()
    print("Test samples:", len(x_test))

    # Save labels once. The FGSM/BIM files for every seed
    # correspond to exactly this same test-set ordering.
    save_labels(y_test)

    for seed, model_path in enumerate(MODEL_PATHS):
        print("\n====================================")
        print(f" MODEL SEED {seed}")
        print("====================================")
        print(f"Loading model: {model_path}")

        model = tf.keras.models.load_model(model_path)
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        generate_adv_for_model(model, x_test, y_test, seed)
        print(f"\nCompleted adversarial generation for seed{seed}.")

    print("\n====================================")
    print(" ALL ADVERSARIAL SAMPLES GENERATED ")
    print("====================================")


if __name__ == "__main__":
    generate_adv()
