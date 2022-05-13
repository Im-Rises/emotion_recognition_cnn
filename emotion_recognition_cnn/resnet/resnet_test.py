import datetime
import os

import numpy as np
from keras import Input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.datasets import cifar10
from keras.layers import (
    Dense,
    BatchNormalization,
    ReLU,
    Conv2D,
    Add,
    AveragePooling2D,
    Flatten,
)
from keras.models import Model, load_model
from tensorflow import Tensor

"""
source : https://www.nablasquared.com/building-a-resnet-in-keras/

input shape : (32,32,3)
layers :
1 conv2D with 64 filters 
2,5,5,2 residual blocks with 64,182,256,512 filters 
averagePooling2D layer with pool size = 4 
flatter layer 
dense layer with 10 output nodes
"""


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(
    x: Tensor, downsample: bool, filters: int, kernel_size: int = 3
) -> Tensor:
    y = Conv2D(
        kernel_size=kernel_size,
        strides=(1 if not downsample else 2),
        filters=filters,
        padding="same",
    )(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size, strides=1, filters=filters, padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1, strides=2, filters=filters, padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


def create_resnet(nbr_labels: int, shape: tuple) -> Model:
    inputs = Input(shape=(32, 32, 3))
    num_filters = 64

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3, strides=1, filters=num_filters, padding="same")(t)
    t = relu_bn(t)

    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters)
        num_filters *= 2

    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(8, activation="softmax")(t)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def create_plain_net() -> Model:
    inputs = Input(shape=(32, 32, 3))
    num_filters = 64

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3, strides=1, filters=num_filters, padding="same")(t)
    t = relu_bn(t)

    num_blocks_list = [4, 10, 10, 4]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            downsample = j == 0 and i != 0
            t = Conv2D(
                kernel_size=3,
                strides=(1 if not downsample else 2),
                filters=num_filters,
                padding="same",
            )(t)
            t = relu_bn(t)
        num_filters *= 2

    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(10, activation="softmax")(t)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    try:
        model = load_model("savedModel")
        model.summary()

    except:
        model = create_resnet()  # or create_plain_net()
        model.summary()

        timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = "cifar-10_res_net_30-" + timestr  # or 'cifar-10_plain_net_30-'+timestr

        checkpoint_path = "checkpoints/" + name + "/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.system("mkdir {}".format(checkpoint_dir))

        # save model after each epoch
        cp_callback = ModelCheckpoint(filepath=checkpoint_path, verbose=1)
        tensorboard_callback = TensorBoard(
            log_dir="tensorboard_logs/" + name, histogram_freq=1
        )

        model.fit(
            x=x_train,
            y=y_train,
            epochs=20,
            verbose=1,
            validation_data=(x_test, y_test),
            batch_size=128,
            callbacks=[cp_callback, tensorboard_callback],
        )
    pred = model.predict(x_test.reshape(None, 32, 32, 3))
    pred = np.argmax(pred, axis=1)[:5]
    label = np.argmax(y_test, axis=1)[:5]

    print(pred)
    print(label)
