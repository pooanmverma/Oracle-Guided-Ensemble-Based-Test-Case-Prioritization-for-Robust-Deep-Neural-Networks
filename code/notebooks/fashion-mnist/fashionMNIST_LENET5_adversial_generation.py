#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fashion-MNIST Adversarial Generation
FGSM + BIM for All Seeds (0–4)
DeepGini-style (White-box)
Saves: images + labels
"""

import os
import numpy as np
import tensorflow as tf


# ================================
# PARAMS
# ================================

BATCH = 256

SEEDS = [0,1,2,3,4,5, 6, 7, 8, 9]

# FGSM (same as DeepGini)
FGSM_EPS_LIST = [0.01, 0.1]

# BIM
BIM_EPS   = 0.01
BIM_ALPHA = 0.002
BIM_STEPS = 10

MODEL_DIR = "models"
SAVE_DIR  = "adv_data_fashion"


os.makedirs(SAVE_DIR, exist_ok=True)



# ================================
# LOAD DATA
# ================================

def load_fashion():

    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_test = x_test.astype("float32") / 255.0
    x_test = x_test[..., np.newaxis]

    y_test = y_test.flatten()

    return x_test, y_test



# ================================
# FGSM (Best Epsilon)
# ================================

def fgsm_best(model, x, y, eps_list):

    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    best_adv  = tf.identity(x)
    best_loss = tf.zeros(len(x))


    for eps in eps_list:

        with tf.GradientTape() as tape:

            tape.watch(x)

            pred = model(x, training=False)

            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y, pred
            )


        grad = tape.gradient(loss, x)

        adv = x + eps * tf.sign(grad)

        adv = tf.clip_by_value(adv, 0, 1)


        adv_pred = model(adv, training=False)

        adv_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y, adv_pred
        )


        better = adv_loss > best_loss

        mask = tf.reshape(better, (-1,1,1,1))

        best_adv  = tf.where(mask, adv, best_adv)
        best_loss = tf.where(better, adv_loss, best_loss)


    return best_adv.numpy()



# ================================
# BIM
# ================================

def bim_attack(model, x, y, eps, alpha, steps):

    x0 = tf.convert_to_tensor(x)
    y  = tf.convert_to_tensor(y)

    x_adv = tf.identity(x0)


    for _ in range(steps):

        with tf.GradientTape() as tape:

            tape.watch(x_adv)

            pred = model(x_adv, training=False)

            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y, pred
            )


        grad = tape.gradient(loss, x_adv)

        x_adv = x_adv + alpha * tf.sign(grad)


        # Projection
        x_adv = tf.clip_by_value(
            x_adv,
            x0 - eps,
            x0 + eps
        )

        x_adv = tf.clip_by_value(x_adv, 0, 1)


    return x_adv.numpy()



# ================================
# MAIN
# ================================

def main():

    print("\n====================================")
    print(" FASHION ADV GENERATION (ALL SEEDS) ")
    print("====================================\n")


    # Load test set once
    x_test, y_test = load_fashion()

    print("Test samples:", len(x_test))


    # Loop over seeds
    for seed in SEEDS:

        print("\n" + "="*50)
        print("Generating for SEED =", seed)
        print("="*50)


        # Reproducibility
        tf.random.set_seed(seed)
        np.random.seed(seed)


        # Load model
        model_path = f"{MODEL_DIR}/fashion_seed{seed}.h5"

        model = tf.keras.models.load_model(model_path)

        print("Loaded:", model_path)


        # ============================
        # FGSM
        # ============================

        print("Running FGSM...")

        fgsm_list = []


        for i in range(0, len(x_test), BATCH):

            xb = x_test[i:i+BATCH]
            yb = y_test[i:i+BATCH]

            adv = fgsm_best(model, xb, yb, FGSM_EPS_LIST)

            fgsm_list.append(adv)


        x_fgsm = np.concatenate(fgsm_list)


        np.save(
            f"{SAVE_DIR}/fashion_fgsm_seed{seed}.npy",
            x_fgsm
        )

        np.save(
            f"{SAVE_DIR}/fashion_fgsm_seed{seed}_label.npy",
            y_test
        )


        print("Saved FGSM:", x_fgsm.shape)


        # ============================
        # BIM
        # ============================

        print("Running BIM...")

        bim_list = []


        for i in range(0, len(x_test), BATCH):

            xb = x_test[i:i+BATCH]
            yb = y_test[i:i+BATCH]

            adv = bim_attack(
                model,
                xb,
                yb,
                BIM_EPS,
                BIM_ALPHA,
                BIM_STEPS
            )

            bim_list.append(adv)


        x_bim = np.concatenate(bim_list)


        np.save(
            f"{SAVE_DIR}/fashion_bim_seed{seed}.npy",
            x_bim
        )

        np.save(
            f"{SAVE_DIR}/fashion_bim_seed{seed}_label.npy",
            y_test
        )


        print("Saved BIM :", x_bim.shape)


        print("Seed", seed, "completed.")


    print("\nAll seeds finished successfully.")



# ================================
# RUN
# ================================

if __name__ == "__main__":

    main()
