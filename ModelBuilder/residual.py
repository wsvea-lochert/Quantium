# This function returns a residual model.
# Written by William Svea-Lochert, Halden, Norway 2021.

from tensorflow import keras
from keras import layers


def residual():
    shape = (224, 224, 3)

    inputs = layers.Input(shape)
    x = inputs

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(416, 3, strides=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(224, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    sizes = [320, 416, 256, 160, 416, 32]

    for size in sizes:
        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(size, 3, padding="same")(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.MaxPooling2D(2, strides=4, padding="same")(x)
    x = layers.SeparableConv2D(30, kernel_size=2, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, strides=4, padding="same")(x)

    outputs = layers.SeparableConv2D(30, kernel_size=1, strides=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.summary()

    return model


    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2])

    # model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(hp_learning_rate), metrics=['mae'])