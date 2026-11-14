"""
CIFAR-10 ResNet-20 Training Script

Trains a ResNet-20 on CIFAR-10 with data augmentation and per-pixel
mean subtraction. Change SEED to train different ensemble members.

Seeds used in the paper:
  SEED=0  -> resnet20_best.h5
  SEED=1  -> resnet20_seed1.h5
  SEED=2  -> resnet20_seed2.h5
  SEED=9  -> resnet20_seed9.h5
  SEED=23 -> resnet20_seed23.h5

Usage:  python cifar10_resnet20_training.py
"""

import tensorflow as tf
import numpy as np
import random
import os

from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

# ================================
# SEED — change for each ensemble member
# ================================
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ================================
# TRAINING PARAMS
# ================================
BATCH_SIZE = 128
EPOCHS = 200
NUM_CLASSES = 10
WEIGHT_DECAY = 1e-4

# ================================
# LOAD CIFAR-10
# ================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test  = y_test.flatten()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Subtract pixel mean (standard CIFAR practice)
mean = np.mean(x_train, axis=0)
x_train -= mean
x_test  -= mean

print("Train:", x_train.shape)
print("Test :", x_test.shape)

# ================================
# LEARNING RATE SCHEDULE
# ================================
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.0005
    elif epoch > 160:
        lr *= 0.001
    elif epoch > 120:
        lr *= 0.01
    elif epoch > 80:
        lr *= 0.1
    return lr

# ================================
# RESNET LAYER
# ================================
def resnet_layer(x, filters, kernel=3, stride=1, activation=True):
    x = layers.Conv2D(
        filters, kernel, strides=stride, padding="same",
        use_bias=False, kernel_regularizer=l2(WEIGHT_DECAY)
    )(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.ReLU()(x)
    return x

# ================================
# RESNET-20
# ================================
def build_resnet20():
    inputs = layers.Input(shape=(32, 32, 3))
    x = resnet_layer(inputs, 16)

    filters = 16
    blocks = 3

    for stage in range(3):
        for block in range(blocks):
            stride = 1
            if stage > 0 and block == 0:
                stride = 2

            y = resnet_layer(x, filters, stride=stride)
            y = resnet_layer(y, filters, activation=False)

            if stage > 0 and block == 0:
                x = resnet_layer(x, filters, kernel=1,
                                 stride=stride, activation=False)

            x = layers.Add()([x, y])
            x = layers.ReLU()(x)

        filters *= 2

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax",
                           kernel_initializer="he_normal")(x)

    return models.Model(inputs, outputs)

# ================================
# BUILD MODEL
# ================================
model = build_resnet20()

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule(0)),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================================
# DATA AUGMENTATION
# ================================
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# ================================
# CALLBACKS
# ================================
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

model_name = f"resnet20_seed{SEED}.h5"

ckpt = callbacks.ModelCheckpoint(
    filepath=os.path.join(save_dir, model_name),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

lr_cb = callbacks.LearningRateScheduler(lr_schedule)

reduce_lr = callbacks.ReduceLROnPlateau(
    factor=0.1, patience=10, min_lr=1e-6
)

cb_list = [ckpt, lr_cb, reduce_lr]

# ================================
# TRAIN
# ================================
print(f"\nTraining ResNet-20 with SEED={SEED}...\n")

model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=cb_list
)

# ================================
# LOAD BEST & EVALUATE
# ================================
model.load_weights(os.path.join(save_dir, model_name))

loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy (seed {SEED}): {round(acc, 4)}")