#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LeNet-1-style Architecture Adapted to MNIST (28x28)
- 5 independent seeds
- Save best model for each seed
- Report test accuracy for each seed
- Report mean and standard deviation
"""

import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K


# ============================================================
# SET SEED
# ============================================================
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# BUILD LE-NET-1-STYLE MODEL ADAPTED TO MNIST
# ============================================================
def build_model(nb_classes=10):

    input_tensor = Input(shape=(28, 28, 1))

    # H1: 12 feature maps, 24x24
    x = Conv2D(
        filters=12,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        activation="tanh"
    )(input_tensor)

    # H2: 12 feature maps, 12x12
    x = Conv2D(
        filters=12,
        kernel_size=(13, 13),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        activation="tanh"
    )(x)

    # H3: 30 fully connected units
    x = Flatten()(x)
    x = Dense(30, activation="tanh")(x)

    # Output: 10 units
    output = Dense(nb_classes, activation="softmax")(x)

    model = Model(
        inputs=input_tensor,
        outputs=output
    )

    return model


# ============================================================
# MAIN
# ============================================================
def model_mnist():

    # --------------------------------------------------------
    # Load MNIST
    # --------------------------------------------------------
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.astype("float32").reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.astype("float32").reshape(-1, 28, 28, 1) / 255.0

    nb_classes = 10

    Y_train = to_categorical(Y_train, nb_classes)
    Y_test = to_categorical(Y_test, nb_classes)

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    # --------------------------------------------------------
    # Model directory
    # --------------------------------------------------------
    os.makedirs("models", exist_ok=True)

    # --------------------------------------------------------
    # Five independent seeds
    # --------------------------------------------------------
    SEEDS = [0, 1, 2, 3, 4]

    results = {}

    # ========================================================
    # TRAIN EACH SEED
    # ========================================================
    for seed in SEEDS:

        print("\n" + "=" * 60)
        print("Training LeNet-1-style Adapted to MNIST with SEED =", seed)
        print("=" * 60)

        K.clear_session()
        set_seed(seed)

        model = build_model(nb_classes)
        model.summary()

        # ----------------------------------------------------
        # SGD configuration
        # ----------------------------------------------------
        optimizer = SGD(
            learning_rate=0.01,
            momentum=0.9,
            nesterov=True
        )

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------
        model_path = f"models/lenet1_mnist_adapted_seed{seed}.h5"

        checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------
        model.fit(
            X_train,
            Y_train,
            batch_size=64,
            epochs=15,
            validation_data=(X_test, Y_test),
            callbacks=[checkpoint],
            verbose=2
        )

        # ----------------------------------------------------
        # Load best model
        # ----------------------------------------------------
        model = load_model(model_path)

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------
        score = model.evaluate(
            X_test,
            Y_test,
            verbose=0
        )

        acc = score[1]
        results[seed] = acc

        print(
            "Seed {} Accuracy: {:.4f}".format(seed, acc)
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS - LE-NET-1-STYLE ADAPTED TO MNIST (5 SEEDS)")
    print("=" * 60)

    for seed, acc in results.items():
        print("Seed {}: {:.4f}".format(seed, acc))

    mean_acc = np.mean(list(results.values()))
    std_acc = np.std(list(results.values()))

    print("\nMean Accuracy: {:.4f}".format(mean_acc))
    print("Std Accuracy : {:.4f}".format(std_acc))


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    model_mnist()