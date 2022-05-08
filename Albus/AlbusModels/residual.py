from tensorflow import keras
from keras import layers


def build_residual_search(hp):
    shape = (224, 224, 3)

    inputs = layers.Input(shape)
    x = inputs

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(hp.Int('units_input', min_value=32, max_value=512, step=32), 3, strides=2, padding="same")(x)

    if hp.Choice("BatchNorm1", [True, False]):  # Testing with and without batch normalization.
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(hp.Int('units_2', min_value=32, max_value=512, step=32), 3, padding="same")(x)

    if hp.Choice("BatchNorm2", [True, False]):  # Testing with and without batch normalization.
        x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for i in range(hp.Int('layers', 2, 6)):
        size = hp.Int(f'unit_block-{str(i)}', min_value=32, max_value=512, step=32)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)

        if hp.Choice("BatchNorm3", [True, False]):  # Testing with and without batch normalization.
            x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)

        if hp.Choice("BatchNorm4", [True, False]):  # Testing with and without batch normalization.
            x = layers.BatchNormalization()(x)

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

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(hp_learning_rate), metrics=['mae'])

    return model
