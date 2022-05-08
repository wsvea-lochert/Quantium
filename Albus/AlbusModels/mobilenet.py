from tensorflow import keras
from keras import layers


def build_mobilenet_search(hp):
    # Load the pre-trained weights of MobileNetV2 and freeze the weights
    backbone = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    backbone.trainable = False

    inputs = layers.Input((224, 224, 3))
    model = keras.applications.mobilenet_v2.preprocess_input(inputs)
    model = backbone(model)

    for i in range(hp.Int('Conv_layers', 0, 6)):
        model = layers.Conv2D(hp.Int('units_' + str(i), min_value=32, max_value=128, step=32), 3, padding="same", name="conv_layer_" + str(i))(model)

        if hp.Choice("BatchNorm" + str(i), [True, False]):  # Testing with and without batch normalization.
            model = layers.BatchNormalization()(model)

        model = layers.Activation("relu")(model)

        if hp.Choice("conv_dropout_C" + str(i), [True, False]):  # Testing with and without dropout.
            model = layers.Dropout(hp.Choice("conv_dropout_" + str(i), values=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), name="conv_dropout_" + str(i))(model)

    model = layers.MaxPooling2D(2, strides=1, padding="same")(model)

    for j in range(hp.Int('separable_layers', 0, 6)):
        model = layers.SeparableConv2D(hp.Int('sep_units_' + str(j), min_value=32, max_value=128, step=32), 3, padding="same", name="separable_layer_" + str(j))(model)

        if hp.Choice("sep_BatchNorm" + str(j), [True, False]):  # Testing with and without batch normalization.
            model = layers.BatchNormalization()(model)

        model = layers.Activation("relu")(model)
        if hp.Choice("sep_dropout_C" + str(j), [True, False]):  # Testing with and without dropout.
            model = layers.Dropout(hp.Choice("sep_dropout_" + str(j), values=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), name="sep_dropout_" + str(j))(model)

        # model = layers.MaxPooling2D(3, strides=3, padding="same")(model)

    model = layers.Dropout(0.2)(model)
    model = layers.SeparableConv2D(
        30, kernel_size=5, strides=1, activation="relu")(model)
    outputs = layers.SeparableConv2D(30, kernel_size=3, strides=1, activation="sigmoid")(model)

    model = keras.Model(inputs, outputs, name="keypoint_detector")

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(hp_learning_rate), metrics=['mae'])
    model.summary()
    return model
