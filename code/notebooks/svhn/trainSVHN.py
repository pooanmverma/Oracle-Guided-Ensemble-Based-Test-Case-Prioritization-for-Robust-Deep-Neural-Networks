'''
SVHN training with 5 seeds
'''

import os
import random
import numpy as np
import tensorflow as tf
import SVNH_DatasetUtil
from keras import Model, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint




# =========================
# set seed
# =========================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================
# model
# =========================

def build_model():

    nb_classes = 10

    input_tensor = Input((32,32,3))

    x = Conv2D(
        filters=6,
        kernel_size=(5,5),
        padding='valid',
        use_bias=False
    )(input_tensor)

    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(
        filters=16,
        kernel_size=(5,5),
        padding='valid',
        use_bias=False
    )(x)

    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Flatten()(x)

    x = Dense(120, activation='relu')(x)

    x = Dense(84, activation='relu')(x)

    output = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)

    sgd = SGD(
        learning_rate=0.01,
        momentum=0.9,
        nesterov=True
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )

    return model


# =========================
# train one seed
# =========================

def train_seed(seed):

    print("\n====================")
    print("SEED", seed)
    print("====================")

    set_seed(seed)

    (X_train, Y_train), (X_test, Y_test) = \
        SVNH_DatasetUtil.load_data()

    model = build_model()

    save_path = f"./model/svhn_seed{seed}.h5"

    checkpoint = ModelCheckpoint(
        filepath=save_path,
        monitor='val_accuracy',
        save_best_only=True
    )

    model.fit(
        X_train,
        Y_train,
        batch_size=64,
        epochs=15,
        validation_data=(X_test, Y_test),
        callbacks=[checkpoint],
        verbose=2
    )

    model = load_model(save_path)

    score = model.evaluate(
        X_test,
        Y_test,
        verbose=0
    )

    print("Test:", score)


# =========================
# main
# =========================

if __name__ == "__main__":

    seeds = [0,1,2,3,4]

    for s in seeds:

        train_seed(s)