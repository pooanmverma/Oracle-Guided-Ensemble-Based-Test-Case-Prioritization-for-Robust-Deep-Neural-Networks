#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# MNIST ADVERSARIAL GENERATION
# LeNet-1-style adapted models
# FGSM (Best-of-Eps) + BIM
# 5 independently trained seeds
# =====================================

import tensorflow as tf
import numpy as np
import os


# =====================================
# PARAMETERS
# =====================================

BATCH = 256

# FGSM
FGSM_EPS_LIST = [0.01, 0.1]

# BIM
BIM_EPS = 0.01
BIM_ALPHA = 0.002
BIM_STEPS = 10

# Five trained models
SEEDS = [0, 1, 2, 3, 4]

# Model directory
MODEL_DIR = "models"

# Adversarial data base directory
SAVE_DIR = "adv_data"


# =====================================
# FGSM (BEST-OF-EPS)
# =====================================

def fgsm_attack_best(model, x, y, eps_list):

    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    best_adv = tf.identity(x)
    best_loss = tf.zeros(
        len(x),
        dtype=tf.float32
    )

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

        # FGSM perturbation
        adv = (
            x +
            eps * tf.sign(grad)
        )

        # Keep image values in [0, 1]
        adv = tf.clip_by_value(
            adv,
            0.0,
            1.0
        )

        # Evaluate adversarial loss
        adv_pred = model(
            adv,
            training=False
        )

        adv_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y,
            adv_pred
        )

        # Select the stronger perturbation
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
# BIM / ITERATIVE FGSM
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

        # Iterative FGSM update
        x_adv = (
            x_adv +
            alpha * tf.sign(grad)
        )

        # Project into epsilon ball
        x_adv = tf.clip_by_value(
            x_adv,
            x0 - eps,
            x0 + eps
        )

        # Keep image values in [0, 1]
        x_adv = tf.clip_by_value(
            x_adv,
            0.0,
            1.0
        )

    return x_adv.numpy()


# =====================================
# LOAD MNIST
# =====================================

def load_mnist():

    (_, _), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize to [0, 1]
    x_test = (
        x_test.astype("float32") /
        255.0
    )

    # 28x28 -> 28x28x1
    x_test = x_test.reshape(
        -1,
        28,
        28,
        1
    )

    y_test = y_test.flatten()

    return x_test, y_test


# =====================================
# GENERATE ADVERSARIAL DATA FOR ONE SEED
# =====================================

def generate_adv_for_seed(
    model,
    x_test,
    y_test,
    seed
):

    # ---------------------------------
    # Separate directory for each seed
    # ---------------------------------
    seed_save_dir = os.path.join(
        SAVE_DIR,
        f"lenet1_seed{seed}"
    )

    os.makedirs(
        seed_save_dir,
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print(
        f"Generating adversarial data for SEED {seed}"
    )
    print("=" * 60)

    # =================================
    # FGSM
    # =================================

    print("\nRunning FGSM (best-of-eps)...")

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

        adv_fgsm.append(adv)

        print(
            f"FGSM: {min(i + BATCH, len(x_test))}"
            f"/{len(x_test)}",
            end="\r"
        )

    print()

    adv_fgsm = np.concatenate(
        adv_fgsm,
        axis=0
    )

    # ---------------------------------
    # Save FGSM
    # ---------------------------------

    fgsm_image_path = os.path.join(
        seed_save_dir,
        "mnist_fgsm_image.npy"
    )

    fgsm_label_path = os.path.join(
        seed_save_dir,
        "mnist_fgsm_label.npy"
    )

    np.save(
        fgsm_image_path,
        adv_fgsm
    )

    np.save(
        fgsm_label_path,
        y_test
    )

    print(
        "FGSM saved:",
        fgsm_image_path
    )

    print(
        "FGSM shape:",
        adv_fgsm.shape
    )


    # =================================
    # BIM
    # =================================

    print("\nRunning BIM...")

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

        adv_bim.append(adv)

        print(
            f"BIM: {min(i + BATCH, len(x_test))}"
            f"/{len(x_test)}",
            end="\r"
        )

    print()

    adv_bim = np.concatenate(
        adv_bim,
        axis=0
    )

    # ---------------------------------
    # Save BIM
    # ---------------------------------

    bim_image_path = os.path.join(
        seed_save_dir,
        "mnist_bim_image.npy"
    )

    bim_label_path = os.path.join(
        seed_save_dir,
        "mnist_bim_label.npy"
    )

    np.save(
        bim_image_path,
        adv_bim
    )

    np.save(
        bim_label_path,
        y_test
    )

    print(
        "BIM saved:",
        bim_image_path
    )

    print(
        "BIM shape:",
        adv_bim.shape
    )

    print(
        f"\nSeed {seed} completed successfully."
    )


# =====================================
# GENERATE FOR ALL FIVE SEEDS
# =====================================

def generate_all_adversarial_data():

    print("\n" + "=" * 60)
    print(" MNIST ADVERSARIAL GENERATION")
    print(" LeNet-1-style Adapted Models")
    print(" FGSM + BIM | 5 Seeds")
    print("=" * 60)

    # ---------------------------------
    # Load MNIST
    # ---------------------------------
    x_test, y_test = load_mnist()

    print(
        "\nTest samples:",
        len(x_test)
    )

    print(
        "Input shape:",
        x_test.shape
    )

    # =================================
    # LOOP OVER FIVE MODELS
    # =================================

    for seed in SEEDS:

        # ---------------------------------
        # Model path
        # ---------------------------------
        model_path = os.path.join(
            MODEL_DIR,
            f"lenet1_mnist_adapted_seed{seed}.h5"
        )

        print("\n" + "-" * 60)
        print(
            f"Loading model for seed {seed}:"
        )
        print(
            model_path
        )

        # ---------------------------------
        # Load model
        # ---------------------------------
        if not os.path.exists(model_path):

            print(
                f"WARNING: Model not found for seed {seed}."
            )

            print(
                "Skipping this seed."
            )

            continue

        model = tf.keras.models.load_model(
            model_path
        )

        print(
            f"Model seed {seed} loaded successfully."
        )

        # ---------------------------------
        # Compile
        # ---------------------------------
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        # ---------------------------------
        # Generate FGSM + BIM
        # ---------------------------------
        generate_adv_for_seed(
            model,
            x_test,
            y_test,
            seed
        )

        # ---------------------------------
        # Clear model/session
        # ---------------------------------
        del model
        tf.keras.backend.clear_session()


    # =================================
    # FINISHED
    # =================================

    print("\n" + "=" * 60)
    print(
        "ALL FIVE SEEDS COMPLETED"
    )
    print("=" * 60)

    for seed in SEEDS:

        print(
            f"\nSeed {seed}:"
        )

        print(
            f"  adv_data/lenet1_seed{seed}/"
        )

        print(
            "    mnist_fgsm_image.npy"
        )

        print(
            "    mnist_fgsm_label.npy"
        )

        print(
            "    mnist_bim_image.npy"
        )

        print(
            "    mnist_bim_label.npy"
        )


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    generate_all_adversarial_data()