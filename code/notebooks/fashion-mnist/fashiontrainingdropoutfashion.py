#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fashion-MNIST CNN Training (Multi-Seed, Modern Keras)

- Python 3.12 compatible
- Uses tf.keras.datasets
- Saves best models
- Reports mean ± std accuracy
"""

# ================================
# IMPORTS
# ================================
import os
import time
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Dense, Activation, Flatten, Dropout
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical


# ================================
# SET SEED
# ================================
def set_seed(seed=0):

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# ================================
# BUILD MODEL
# ================================
def build_model(num_classes=10):

    inputs = Input(shape=(28, 28, 1))

    x = Conv2D(6, (5,5), padding="valid", use_bias=False)(inputs)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(16, (5,5), padding="valid", use_bias=False)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Flatten()(x)

    x = Dense(120, activation="relu")(x)
    x = Dense(84, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


# ================================
# LOAD DATA
# ================================
def load_data():

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0

    # Add channel dimension
    X_train = np.expand_dims(X_train, -1)
    X_test  = np.expand_dims(X_test, -1)

    # One-hot
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test, 10)

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    return X_train, y_train, X_test, y_test


# ================================
# TRAIN ONE SEED
# ================================
def train_one_seed(seed, X_train, y_train, X_test, y_test):

    print("\n" + "="*50)
    print(f"Training with SEED = {seed}")
    print("="*50)

    # Reproducibility
    set_seed(seed)
    K.clear_session()


    # Build
    model = build_model()
    model.summary()


    # Compile
    optimizer = SGD(
        learning_rate=0.01,
        decay=1e-6,
        momentum=0.9,
        nesterov=True
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


    # Save dir
    os.makedirs("models", exist_ok=True)

    model_path = f"models/fashion_drop_seed{seed}.h5"


    # Callback
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )


    # Train
    model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=15,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint],
        verbose=2
    )


    # Load best
    best_model = load_model(model_path)


    # Evaluate
    loss, acc = best_model.evaluate(X_test, y_test, verbose=0)

    print(f"Seed {seed} Accuracy: {acc:.4f}")

    return acc


# ================================
# MAIN
# ================================
def main():

    start = time.time()

    # Load once
    X_train, y_train, X_test, y_test = load_data()


    # Seeds
    SEEDS = [0,1,2,3,4,5]

    results = {}


    # Train
    for seed in SEEDS:

        acc = train_one_seed(
            seed,
            X_train, y_train,
            X_test, y_test
        )

        results[seed] = acc


    # ============================
    # REPORT
    # ============================
    print("\n" + "="*50)
    print("FINAL RESULTS (5 SEEDS)")
    print("="*50)

    for s, a in results.items():
        print(f"Seed {s}: {a:.4f}")

    mean_acc = np.mean(list(results.values()))
    std_acc  = np.std(list(results.values()))

    print("\nMean Accuracy:", round(mean_acc,4))
    print("Std  Accuracy:", round(std_acc,4))

    print("\nTotal Time:", round(time.time()-start,2), "seconds")


# ================================
# RUN
# ================================
if __name__ == "__main__":

    main()