from tensorflow import keras
from keras import layers


def build_cnn_search(hp):
    input_shape = (224, 224, 3)

    input_layer = layers.Input(input_shape)

    x = input_layer
    x = layers.Conv2D(hp.Int('units_Input', min_value=96, max_value=512, step=32), 3, padding="same", name="first_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, strides=2, padding="same")(x)

    for i in range(hp.Int('Conv_layers', 2, 6)):
        x = layers.Conv2D(hp.Int('units_' + str(i), min_value=96, max_value=512, step=32), 3, padding="same", name="conv_layer_"+str(i))(x)

        if hp.Choice("BatchNorm" + str(i), [True, False]):  # Testing with and without batch normalization.
            x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)

        if hp.Choice("conv_dropout_C" + str(i), [True, False]):  # Testing with and without dropout.
            x = layers.Dropout(hp.Choice("conv_dropout_" + str(i), values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), name="conv_dropout_" + str(i))(x)

    x = layers.MaxPooling2D(2, strides=2, padding="same")(x)

    for j in range(hp.Int('separable_layers', 2, 6)):
        x = layers.SeparableConv2D(hp.Int('sep_units_'+str(j), min_value=96, max_value=512, step=32), 3, padding="same", name="separable_layer_"+str(j))(x)

        if hp.Choice("BatchNorm" + str(j), [True, False]):  # Testing with and without batch normalization.
            x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        if hp.Choice("sep_dropout_C" + str(j), [True, False]):  # Testing with and without dropout.
            x = layers.Dropout(hp.Choice("sep_dropout_" + str(j), values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), name="sep_dropout_" + str(j))(x)

    x = layers.MaxPooling2D(3, strides=3, padding="same")(x)

    x = layers.SeparableConv2D(hp.Int('sep_out_units_0', min_value=96, max_value=512, step=32), kernel_size=5, strides=1, activation="relu", name="sep_out_0")(x)
    x = layers.MaxPooling2D(2, strides=2, padding="same")(x)
    x = layers.SeparableConv2D(hp.Int('sep_out_units_1', min_value=96, max_value=512, step=32), kernel_size=3, strides=1, activation="relu", name="sep_out_1")(x)
    x = layers.MaxPooling2D(2, strides=2, padding="same")(x)
    x = layers.SeparableConv2D(hp.Int('sep_out_units_2', min_value=96, max_value=512, step=32), kernel_size=2, strides=1, activation="relu", name="sep_out_2")(x)
    x = layers.MaxPooling2D(2, strides=2, padding="same")(x)

    outputs = layers.SeparableConv2D(30, kernel_size=1, strides=1, activation="sigmoid", name="sep_output")(x)
    # outputs = layers.Flatten()(x)

    model = keras.Model(input_layer, outputs)
    model.summary()

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(hp_learning_rate), metrics=['mae'])
    return model
