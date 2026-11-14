#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
MNIST Multi-Seed Training Program

- 5 seeds
- Save best model per seed
- Report accuracy
'''

# ================================
# IMPORTS
# ================================
import os
import time
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K


# ================================
# SET SEED
# ================================
def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# ================================
# BUILD MODEL
# ================================
def build_model(nb_classes):

    input_tensor = Input((28,28,1))

    temp = Conv2D(6, (5,5), padding='valid', use_bias=False)(input_tensor)
    temp = Activation('relu')(temp)

    temp = MaxPooling2D(pool_size=(2,2))(temp)

    temp = Conv2D(16, (5,5), padding='valid', use_bias=False)(temp)
    temp = Activation('relu')(temp)

    temp = MaxPooling2D(pool_size=(2,2))(temp)

    temp = Flatten()(temp)

    temp = Dense(120, activation='relu')(temp)
    temp = Dense(84, activation='relu')(temp)

    output = Dense(nb_classes, activation='softmax')(temp)

    model = Model(inputs=input_tensor, outputs=output)

    return model


# ================================
# MAIN FUNCTION
# ================================
def model_mnist():

    # ----------------
    # Load Data
    # ----------------
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.astype('float32').reshape(-1,28,28,1) / 255.0
    X_test  = X_test.astype('float32').reshape(-1,28,28,1) / 255.0

    print('Train:{}, Test:{}'.format(len(X_train), len(X_test)))

    nb_classes = 10

    Y_train = to_categorical(Y_train, nb_classes)
    Y_test  = to_categorical(Y_test, nb_classes)

    print('Data loaded successfully')


    # ----------------
    # Create Folder
    # ----------------
    os.makedirs("models", exist_ok=True)


    # ----------------
    # Seeds
    # ----------------
    SEEDS = [0, 1, 2, 3, 4]

    results = {}


    # ================================
    # MULTI-SEED LOOP
    # ================================
    for seed in SEEDS:

        print("\n" + "="*50)
        print("Training with SEED =", seed)
        print("="*50)


        # ----------------
        # Set Seed
        # ----------------
        set_seed(seed)
        K.clear_session()


        # ----------------
        # Build Model
        # ----------------
        model = build_model(nb_classes)

        model.summary()


        # ----------------
        # Compile
        # ----------------
        sgd = SGD(
            learning_rate=0.01,
            decay=1e-6,
            momentum=0.9,
            nesterov=True
        )

        model.compile(
            loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy']
        )


        # ----------------
        # Checkpoint
        # ----------------
        model_path = f"models/model_mnist_seed{seed}.h5"

        checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )


        # ----------------
        # Train
        # ----------------
        model.fit(
            X_train,
            Y_train,
            batch_size=64,
            epochs=15,
            validation_data=(X_test, Y_test),
            callbacks=[checkpoint],
            verbose=2
        )


        # ----------------
        # Load Best Model
        # ----------------
        model = load_model(model_path)


        # ----------------
        # Evaluate
        # ----------------
        score = model.evaluate(X_test, Y_test, verbose=0)

        acc = score[1]

        results[seed] = acc

        print("Seed {} Accuracy: {:.4f}".format(seed, acc))


    # ================================
    # FINAL REPORT
    # ================================
    print("\n" + "="*50)
    print("FINAL RESULTS (5 SEEDS)")
    print("="*50)

    for seed, acc in results.items():
        print("Seed {}: {:.4f}".format(seed, acc))

    mean_acc = np.mean(list(results.values()))
    std_acc  = np.std(list(results.values()))

    print("\nMean Accuracy: {:.4f}".format(mean_acc))
    print("Std  Accuracy: {:.4f}".format(std_acc))


# ================================
# RUN
# ================================
if __name__ == '__main__':

    model_mnist()
