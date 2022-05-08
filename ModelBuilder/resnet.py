from tensorflow import keras
from keras import layers
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def resnet():
    # Load the pre-trained weights of MobileNetV2 and freeze the weights
    backbone = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    backbone.trainable = True
    print("Number of layers in the base model: ", len(backbone.layers))
    fine_tune_at = 100

    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = layers.Input((224, 224, 3))
    model = keras.applications.resnet50.preprocess_input(inputs)
    model = backbone(model)

    model = layers.Conv2D(96, 3, padding="same", name="conv_layer_0")(model)
    model = layers.BatchNormalization()(model)
    model = layers.Activation("relu")(model)

    model = layers.Conv2D(64, 3, padding="same", name="conv_layer_1")(model)
    model = layers.Activation("relu")(model)

    model = layers.Conv2D(96, 3, padding="same", name="conv_layer_2")(model)
    model = layers.BatchNormalization()(model)
    model = layers.Activation("relu")(model)

    model = layers.MaxPooling2D(2, strides=1, padding="same")(model)

    model = layers.SeparableConv2D(64, 3, padding="same", name="separable_layer_0")(model)
    model = layers.Activation("relu")(model)

    model = layers.Dropout(0.2)(model)
    model = layers.SeparableConv2D(
        30, kernel_size=5, strides=1, activation="relu")(model)
    outputs = layers.SeparableConv2D(30, kernel_size=3, strides=1, activation="sigmoid")(model)

    model = keras.Model(inputs, outputs, name="keypoint_detector")

    # model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(0.001), metrics=['mse', 'mae'])
    # model.summary()
    return model
