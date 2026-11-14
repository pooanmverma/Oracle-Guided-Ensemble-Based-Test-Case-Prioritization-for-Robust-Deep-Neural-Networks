#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# FASHION-MNIST ADVERSARIAL GENERATION
# FOR 5 MC-DROPOUT MODELS
#
# FGSM (Best-of-Eps) + BIM (Iter-FGSM)
# Pure TensorFlow
#
# Each independently trained MC-Dropout model is used
# as the white-box attack target separately.
#
# Models:
#   fashion_drop_seed0.h5
#   fashion_drop_seed1.h5
#   fashion_drop_seed2.h5
#   fashion_drop_seed3.h5
#   fashion_drop_seed4.h5
#
# Outputs:
#   adv_data_fashion/fashion_drop_fgsm_seed0.npy ... seed4.npy
#   adv_data_fashion/fashion_drop_bim_seed0.npy  ... seed4.npy
#
# The original Fashion-MNIST y_test is used during attack
# generation, but label files are not saved because the
# evaluation pipeline can reconstruct y_test in the same
# test-set ordering.
# =====================================

import os
import tensorflow as tf
import numpy as np


# =====================================
# PARAMS
# =====================================

BATCH = 256

# FGSM epsilons
FGSM_EPS_LIST = [0.01, 0.1]

# BIM (Iter-FGSM)
BIM_EPS = 0.01
BIM_ALPHA = 0.002
BIM_STEPS = 10

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

SAVE_DIR = f"{BASE}/adv_data_fashion"

MODEL_PATHS = [
    f"{BASE}/models/fashion_drop_seed0.h5",
    f"{BASE}/models/fashion_drop_seed1.h5",
    f"{BASE}/models/fashion_drop_seed2.h5",
    f"{BASE}/models/fashion_drop_seed3.h5",
    f"{BASE}/models/fashion_drop_seed4.h5",
]

SEEDS = [0, 1, 2, 3, 4]

os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================
# FGSM (BEST OF MULTI-EPS)
# =====================================

def fgsm_attack_best(model, x, y, eps_list):

    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    best_adv = tf.identity(x)
    best_loss = tf.zeros(len(x))

    for eps in eps_list:

        with tf.GradientTape() as tape:

            tape.watch(x)

            pred = model(
                x,
                training=False
            )

            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y,
                pred
            )

        grad = tape.gradient(
            loss,
            x
        )

        adv = x + eps * tf.sign(grad)

        adv = tf.clip_by_value(
            adv,
            0,
            1
        )

        # Loss on adversarial samples
        adv_pred = model(
            adv,
            training=False
        )

        adv_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y,
            adv_pred
        )

        # Keep stronger attack
        better = adv_loss > best_loss

        mask = tf.reshape(
            better,
            (-1, 1, 1, 1)
        )

        best_adv = tf.where(
            mask,
            adv,
            best_adv
        )

        best_loss = tf.where(
            better,
            adv_loss,
            best_loss
        )

    return best_adv.numpy()


# =====================================
# BIM (ITER-FGSM)
# =====================================

def bim_attack(
    model,
    x,
    y,
    eps,
    alpha,
    steps
):

    x0 = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    x_adv = tf.identity(x0)

    for _ in range(steps):

        with tf.GradientTape() as tape:

            tape.watch(x_adv)

            pred = model(
                x_adv,
                training=False
            )

            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y,
                pred
            )

        grad = tape.gradient(
            loss,
            x_adv
        )

        x_adv = (
            x_adv
            + alpha * tf.sign(grad)
        )

        # Project to eps ball
        x_adv = tf.clip_by_value(
            x_adv,
            x0 - eps,
            x0 + eps
        )

        # Keep valid image range
        x_adv = tf.clip_by_value(
            x_adv,
            0,
            1
        )

    return x_adv.numpy()


# =====================================
# LOAD FASHION-MNIST
# =====================================

def load_fashion():

    (_, _), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    x_test = (
        x_test.astype("float32")
        / 255.0
    )

    x_test = np.expand_dims(
        x_test,
        -1
    )

    y_test = y_test.astype(
        "int64"
    )

    return x_test, y_test


# =====================================
# GENERATE ADV DATA FOR ONE MODEL
# =====================================

def generate_adv_for_model(
    model,
    x_test,
    y_test,
    seed
):

    # =================================
    # FGSM
    # =================================

    print(
        f"\nRunning FGSM for MC-Dropout seed{seed}..."
    )

    adv_fgsm = []

    for i in range(
        0,
        len(x_test),
        BATCH
    ):

        x_batch = x_test[
            i:i + BATCH
        ]

        y_batch = y_test[
            i:i + BATCH
        ]

        adv = fgsm_attack_best(
            model,
            x_batch,
            y_batch,
            FGSM_EPS_LIST
        )

        adv_fgsm.append(
            adv
        )

    adv_fgsm = np.concatenate(
        adv_fgsm,
        axis=0
    )

    fgsm_path = (
        f"{SAVE_DIR}/"
        f"fashion_drop_fgsm_seed{seed}.npy"
    )

    np.save(
        fgsm_path,
        adv_fgsm
    )

    print(
        f"FGSM saved: "
        f"{fgsm_path} "
        f"{adv_fgsm.shape}"
    )


    # =================================
    # BIM
    # =================================

    print(
        f"\nRunning BIM for MC-Dropout seed{seed}..."
    )

    adv_bim = []

    for i in range(
        0,
        len(x_test),
        BATCH
    ):

        x_batch = x_test[
            i:i + BATCH
        ]

        y_batch = y_test[
            i:i + BATCH
        ]

        adv = bim_attack(
            model,
            x_batch,
            y_batch,
            BIM_EPS,
            BIM_ALPHA,
            BIM_STEPS
        )

        adv_bim.append(
            adv
        )

    adv_bim = np.concatenate(
        adv_bim,
        axis=0
    )

    bim_path = (
        f"{SAVE_DIR}/"
        f"fashion_drop_bim_seed{seed}.npy"
    )

    np.save(
        bim_path,
        adv_bim
    )

    print(
        f"BIM saved: "
        f"{bim_path} "
        f"{adv_bim.shape}"
    )


# =====================================
# MAIN
# =====================================

def main():

    print(
        "\n=============================================="
    )

    print(
        " FASHION-MNIST MC-DROPOUT ADV GENERATION "
    )

    print(
        "==============================================\n"
    )

    x_test, y_test = load_fashion()

    print(
        f"Test samples: {len(x_test)}"
    )

    # =================================
    # LOOP OVER FIVE DROPOUT MODELS
    # =================================

    for seed, model_path in zip(
        SEEDS,
        MODEL_PATHS
    ):

        print(
            "\n=============================================="
        )

        print(
            f" MC-DROPOUT MODEL SEED {seed}"
        )

        print(
            "=============================================="
        )

        if not os.path.isfile(
            model_path
        ):
            raise FileNotFoundError(
                f"\nModel not found:\n{model_path}"
            )

        print(
            f"Loading model: {model_path}"
        )

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        # IMPORTANT:
        # This model itself is the white-box attack target.
        generate_adv_for_model(
            model,
            x_test,
            y_test,
            seed
        )

        del model
        tf.keras.backend.clear_session()

        print(
            f"\nCompleted adversarial generation "
            f"for MC-Dropout seed{seed}."
        )


    print(
        "\n=============================================="
    )

    print(
        " ALL 5 MC-DROPOUT ADVERSARIAL SAMPLES GENERATED "
    )

    print(
        "=============================================="
    )


# =====================================
# RUN
# =====================================

if __name__ == "__main__":
    main()
