#!/usr/bin/env python3
# =====================================
# SVHN ADVERSARIAL GENERATION (PER SEED)
# FGSM + BIM
# =====================================

import tensorflow as tf
import numpy as np
import os

import SVNH_DatasetUtil


# =====================================
# PARAMS
# =====================================

BATCH = 256

SEEDS = [0,1,2,3,4]

FGSM_EPS_LIST = [0.01, 0.1]

BIM_EPS   = 0.01
BIM_ALPHA = 0.002
BIM_STEPS = 10

SAVE_DIR = "adv_data_svhn"

MODEL_DIR = "model"

os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================
# FGSM
# =====================================

def fgsm_attack_best(model, x, y, eps_list):

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


# =====================================
# BIM
# =====================================

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

        x_adv = tf.clip_by_value(
            x_adv,
            x0 - eps,
            x0 + eps
        )

        x_adv = tf.clip_by_value(x_adv, 0, 1)

    return x_adv.numpy()


# =====================================
# LOAD SVHN
# =====================================

def load_svhn():

    (_, _), (x_test, y_test) = SVNH_DatasetUtil.load_data()

    y_test = np.argmax(y_test, axis=1)

    return x_test, y_test


# =====================================
# GENERATE FOR ONE SEED
# =====================================

def generate_for_seed(seed, x_test, y_test):

    print("\n========================")
    print("SEED", seed)
    print("========================")

    model_path = f"{MODEL_DIR}/svhn_seed{seed}.h5"

    model = tf.keras.models.load_model(model_path)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )


    # ---------- FGSM ----------

    adv_fgsm = []

    for i in range(0, len(x_test), BATCH):

        x_batch = x_test[i:i+BATCH]
        y_batch = y_test[i:i+BATCH]

        adv = fgsm_attack_best(
            model,
            x_batch,
            y_batch,
            FGSM_EPS_LIST
        )

        adv_fgsm.append(adv)

    adv_fgsm = np.concatenate(adv_fgsm, axis=0)

    np.save(
        f"{SAVE_DIR}/svhn_fgsm_{seed}.npy",
        adv_fgsm
    )

    print("FGSM saved", adv_fgsm.shape)


    # ---------- BIM ----------

    adv_bim = []

    for i in range(0, len(x_test), BATCH):

        x_batch = x_test[i:i+BATCH]
        y_batch = y_test[i:i+BATCH]

        adv = bim_attack(
            model,
            x_batch,
            y_batch,
            BIM_EPS,
            BIM_ALPHA,
            BIM_STEPS
        )

        adv_bim.append(adv)

    adv_bim = np.concatenate(adv_bim, axis=0)

    np.save(
        f"{SAVE_DIR}/svhn_bim_{seed}.npy",
        adv_bim
    )

    print("BIM saved", adv_bim.shape)


# =====================================
# MAIN
# =====================================

def main():

    print("\nSVHN ADV PER SEED\n")

    x_test, y_test = load_svhn()

    print("Samples:", len(x_test))

    for s in SEEDS:

        generate_for_seed(
            s,
            x_test,
            y_test
        )


if __name__ == "__main__":

    main()