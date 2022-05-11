import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model, load_model
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Input,
    Add,
    Dense,
    Activation,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    GlobalMaxPooling2D,
    MaxPool2D,
)
from keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform

from emotion_recognition_cnn.resnet.DataLoader import DataLoader
from handle_dataset import (
    create_dictionary,
    preprocess_images,
    load_dataset,
    show_image,
)


def identity_block(x: list, f, filters, stage, block):
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"
    F1, F2, F3 = filters

    x_shortcut = x

    x = Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=conv_name_base + "2a",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters=F2,
        kernel_size=(f, f),
        strides=(1, 1),
        padding="same",
        name=conv_name_base + "2b",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=conv_name_base + "2c",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=bn_name_base + "2c")(x)

    x = Add()([x, x_shortcut])  # SKIP Connection
    x = Activation("relu")(x)

    return x


def convolutional_block(x, f, filters, stage, block, s=2):
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    f1, f2, f3 = filters

    x_shortcut = x

    x = Conv2D(
        filters=f1,
        kernel_size=(1, 1),
        strides=(s, s),
        padding="valid",
        name=conv_name_base + "2a",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters=f2,
        kernel_size=(f, f),
        strides=(1, 1),
        padding="same",
        name=conv_name_base + "2b",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(
        filters=f3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=conv_name_base + "2c",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name=bn_name_base + "2c")(x)

    x_shortcut = Conv2D(
        filters=f3,
        kernel_size=(1, 1),
        strides=(s, s),
        padding="valid",
        name=conv_name_base + "1",
        kernel_initializer=glorot_uniform(seed=0),
    )(x_shortcut)
    x_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(x_shortcut)

    x = Add()([x, x_shortcut])
    x = Activation("relu")(x)

    return x


def ResNet50(input_shape: tuple = (224, 224, 3)) -> Model:
    x_input = Input(input_shape)

    x = ZeroPadding2D((3, 3))(x_input)

    x = Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        name="conv1",
        kernel_initializer=glorot_uniform(seed=0),
    )(x)
    x = BatchNormalization(axis=3, name="bn_conv1")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = convolutional_block(x, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b")
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="c")

    x = convolutional_block(x, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="b")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="c")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="d")

    x = convolutional_block(x, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="d")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="e")
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="f")

    x = x = convolutional_block(
        x, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2
    )
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    x = AveragePooling2D(pool_size=(2, 2), padding="same")(x)

    model = Model(inputs=x_input, outputs=x, name="ResNet50")

    return model


if __name__ == "__main__":

    train_path = "./../../../../Databases/FER-2013/train/"
    test_path = "./../../../../Databases/FER-2013/test/"

    fer2013 = DataLoader(
        train_path=train_path, test_path=test_path, labels=[i for i in range(7)]
    )

    x_train, y_train = fer2013.get_train_data()
    x_test, y_test = fer2013.get_test_data()

    # print(x_train[0].shape)
    # print(x_test.shape)

    plt.imshow(x_train[0])
    plt.title(y_train[0])
    plt.show()

    plt.imshow(x_test[0])
    plt.title(y_test[0])
    plt.show()

    # plt.imshow(x_test[0])
    # plt.show()

    """
    base_model = ResNet50(input_shape=(224, 224, 3))

    headModel = base_model.output
    headModel = Flatten()(headModel)
    headModel = Dense(
        256, activation="relu", name="fc1", kernel_initializer=glorot_uniform(seed=0)
    )(headModel)
    headModel = Dense(
        128, activation="relu", name="fc2", kernel_initializer=glorot_uniform(seed=0)
    )(headModel)
    headModel = Dense(
        1, activation="sigmoid", name="fc3", kernel_initializer=glorot_uniform(seed=0)
    )(headModel)

    model = Model(inputs=base_model.input, outputs=headModel)

    base_model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

    for layer in base_model.layers:
        layer.trainable = False

    es = EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=20)

    mc = ModelCheckpoint(
        filename="best_model.h5", filepath="./", monitor="val_accuracy", verbose=1
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    H = model.fit_generator(
        (x_train, y_train),
        validation_data=(x_test, y_test),
        epochs=100,
        verbose=1,
        callbacks=[mc, es],
    )

    model.load_weights("best_model.h5")

    print(model.evaluate_generator((x_test, y_test)))

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    """
