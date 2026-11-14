# =====================================
# CIFAR-10 ADVERSARIAL GENERATION
# FGSM (Best-of-Eps) + BIM (Iter-FGSM)
# FOR ALL 5 RESNET-20 MODELS
# =====================================

import tensorflow as tf
import numpy as np
import os


# =====================================
# PARAMS
# =====================================

BATCH = 256

# FGSM epsilons (DeepGini setting)
FGSM_EPS_LIST = [0.01, 0.1]

# BIM (Iter-FGSM)
BIM_EPS = 0.01
BIM_ALPHA = 0.002
BIM_STEPS = 10

# Adversarial data directory
SAVE_DIR = "adv_data"

os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================
# MODELS
# =====================================

MODELS = {
    "best": "models/resnet20_best.h5",
    "seed1": "models/resnet20_seed1.h5",
    "seed2": "models/resnet20_seed2.h5",
    "seed9": "models/resnet20_seed9.h5",
    "seed23": "models/resnet20_seed23.h5",
}


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

            pred = model(x)

            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y,
                pred
            )

        grad = tape.gradient(loss, x)

        adv = x + eps * tf.sign(grad)

        adv = tf.clip_by_value(
            adv,
            0,
            1
        )

        # Compute adversarial loss
        adv_pred = model(adv)

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
# BIM (ITERATIVE FGSM)
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

            pred = model(x_adv)

            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y,
                pred
            )

        grad = tape.gradient(
            loss,
            x_adv
        )

        x_adv = x_adv + alpha * tf.sign(grad)

        # Project to epsilon ball
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
# LOAD CIFAR-10
# =====================================

def load_cifar10():

    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    y_test = y_test.flatten()

    x_test = x_test.astype("float32") / 255.0

    return x_test, y_test


# =====================================
# GENERATE ADVERSARIAL DATA
# FOR ONE MODEL
# =====================================

def generate_adv_for_model(
    model_name,
    model_path,
    x_test,
    y_test
):

    print("\n")
    print("=" * 60)
    print(f" MODEL = {model_name}")
    print("=" * 60)

    print("Model path:", model_path)

    # =================================
    # LOAD MODEL
    # =================================

    model = tf.keras.models.load_model(
        model_path
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    print("Model loaded successfully.")

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
            f"\rFGSM: {min(i + BATCH, len(x_test))}/{len(x_test)}",
            end=""
        )

    print()

    adv_fgsm = np.concatenate(
        adv_fgsm,
        axis=0
    )

    fgsm_path = os.path.join(
        SAVE_DIR,
        f"cifar10_fgsm_resnet20_{model_name}.npy"
    )

    np.save(
        fgsm_path,
        adv_fgsm
    )

    print(
        "FGSM saved:",
        fgsm_path
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
            f"\rBIM: {min(i + BATCH, len(x_test))}/{len(x_test)}",
            end=""
        )

    print()

    adv_bim = np.concatenate(
        adv_bim,
        axis=0
    )

    bim_path = os.path.join(
        SAVE_DIR,
        f"cifar10_bim_resnet20_{model_name}.npy"
    )

    np.save(
        bim_path,
        adv_bim
    )

    print(
        "BIM saved:",
        bim_path
    )

    print(
        "BIM shape:",
        adv_bim.shape
    )

    # =================================
    # CHECK
    # =================================

    print(
        f"\nCompleted model: {model_name}"
    )


# =====================================
# GENERATE FOR ALL MODELS
# =====================================

def generate_adv():

    print("\n")
    print("=" * 60)
    print(" CIFAR-10 ADVERSARIAL GENERATION")
    print(" FGSM (Best-of-Eps) + BIM")
    print(" ALL 5 RESNET-20 MODELS")
    print("=" * 60)

    # =================================
    # LOAD DATA ONCE
    # =================================

    print("\nLoading CIFAR-10...")

    x_test, y_test = load_cifar10()

    print(
        "Test samples:",
        len(x_test)
    )

    print(
        "Image shape:",
        x_test.shape
    )

    # =================================
    # RUN EACH MODEL
    # =================================

    for model_name, model_path in MODELS.items():

        if not os.path.exists(model_path):

            print(
                f"\n[WARNING] Missing model: {model_path}"
            )

            continue

        generate_adv_for_model(
            model_name,
            model_path,
            x_test,
            y_test
        )

    # =================================
    # FINAL CHECK
    # =================================

    print("\n")
    print("=" * 60)
    print(" ADVERSARIAL GENERATION COMPLETED")
    print("=" * 60)

    print("\nGenerated files:\n")

    for model_name in MODELS:

        fgsm_file = os.path.join(
            SAVE_DIR,
            f"cifar10_fgsm_resnet20_{model_name}.npy"
        )

        bim_file = os.path.join(
            SAVE_DIR,
            f"cifar10_bim_resnet20_{model_name}.npy"
        )

        if os.path.exists(fgsm_file):
            print("OK:", fgsm_file)
        else:
            print("MISSING:", fgsm_file)

        if os.path.exists(bim_file):
            print("OK:", bim_file)
        else:
            print("MISSING:", bim_file)

    print("\nAll adversarial samples generated successfully.")


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    generate_adv()