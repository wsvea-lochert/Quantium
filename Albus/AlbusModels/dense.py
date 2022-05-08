from tensorflow import keras
from keras import layers


def model_builder(hp):

    shape = (224, 224, 3)

    input_layer = layers.Input(shape)
    x = input_layer

    x = layers.Conv2D(hp.Int('units_Input', min_value=16, max_value=128, step=8), 3, padding="same", name="first_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, strides=2, padding="same")(x)

    for i in range(hp.Int('Conv_layers', 2, 6)):
        x = layers.Conv2D(hp.Int('conv_units_' + str(i), min_value=16, max_value=128, step=8), 3, padding="same", name="conv_layer_"+str(i))(x)

        if hp.Choice("BatchNorm" + str(i), [True, False]):  # Testing with and without batch normalization.
            x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)

        if hp.Choice("conv_dropout_C" + str(i), [True, False]):  # Testing with and without dropout.
            x = layers.Dropout(hp.Choice("conv_dropout_" + str(i), values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), name="conv_dropout_" + str(i))(x)

        x = layers.MaxPooling2D(2, strides=2, padding="same")(x)

    for j in range(hp.Int('separable_layers', 2, 6)):
        x = layers.SeparableConv2D(hp.Int('sep_units_'+str(j), min_value=32, max_value=512, step=32), 3, padding="same", name="separable_layer_"+str(j))(x)

        if hp.Choice("sep_BatchNorm" + str(j), [True, False]):  # Testing with and without batch normalization.
            x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        if hp.Choice("sep_dropout_C" + str(j), [True, False]):  # Testing with and without dropout.
            x = layers.Dropout(hp.Choice("sep_dropout_" + str(j), values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), name="sep_dropout_" + str(j))(x)

    x = layers.Flatten()(x)

    for j in range(hp.Int('dense_layers', 1, 4)):
        x = layers.Dense(hp.Int('dense_units_'+str(j), min_value=32, max_value=500, step=32), name="dense_layer_"+str(j))(x)

        if hp.Choice("dense_BatchNorm" + str(j), [True, False]):  # Testing with and without batch normalization.
            x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        if hp.Choice("dense_dropout_C" + str(j), [True, False]):  # Testing with and without dropout.
            x = layers.Dropout(hp.Choice("dense_dropout_" + str(j), values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), name="dense_dropout_" + str(j))(x)

    outputs = layers.Dense(30, activation="sigmoid", name="output")(x)

    model = keras.Model(input_layer, outputs)
    model.summary()

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(hp_learning_rate), metrics=['mae'])
    return model
