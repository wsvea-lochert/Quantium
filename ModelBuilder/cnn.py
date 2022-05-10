from tensorflow import keras
from keras import layers


def cnn():
    shape = (224, 224, 3)

    input_layer = layers.Input(shape)
    x = input_layer

    x = layers.Conv2D(352, 3, padding="same", name="first_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, strides=2, padding="same")(x)

    x = layers.Conv2D(384, 3, padding="same")(x)  # Units_0
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(352, 3, padding="same")(x)  # units_1
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(416, 3, padding="same")(x)  # units_2
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(96, 3, padding="same")(x)  # units_3
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, 3, padding="same")(x)  # units_4
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.MaxPooling2D(2, strides=2, padding="same")(x)

    x = layers.SeparableConv2D(320, 3, padding="same")(x)  # sep_units_0
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.SeparableConv2D(352, 3, padding="same")(x)  # sep_units_1
    x = layers.Activation("relu")(x)

    x = layers.SeparableConv2D(128, 3, padding="same")(x)  # sep_units_2
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.SeparableConv2D(256, 3, padding="same")(x)  # sep_units_3
    x = layers.Activation("relu")(x)

    x = layers.SeparableConv2D(352, 3, padding="same")(x)  # sep_units_4
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.SeparableConv2D(416, 3, padding="same")(x)  # sep_units_5
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(3, strides=3, padding="same")(x)

    x = layers.SeparableConv2D(448, kernel_size=5, strides=1, activation="relu", name="sep_out_0")(x)
    x = layers.MaxPooling2D(2, strides=2, padding="same")(x)
    x = layers.SeparableConv2D(224, kernel_size=3, strides=1, activation="relu", name="sep_out_1")(x)
    x = layers.MaxPooling2D(2, strides=2, padding="same")(x)
    x = layers.SeparableConv2D(480, kernel_size=2, strides=1, activation="relu", name="sep_out_2")(x)
    x = layers.MaxPooling2D(2, strides=2, padding="same")(x)

    outputs = layers.SeparableConv2D(30, kernel_size=1, strides=1, activation="sigmoid", name="output")(x)

    model = keras.Model(input_layer, outputs, name="McFly_cnn_50epochs")
    model.summary()

    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
    return model