import os

import pandas as pd
from keras.layers import Input, Lambda, Dense, Flatten, MaxPooling2D, Conv2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow


def create_model():
    model = tf.keras.Sequential()
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            input_shape=IMSIZE + [1],
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            input_shape=IMSIZE + [1],
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            input_shape=IMSIZE + [1],
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten data
    model.add(Flatten())

    # ANN classic layer
    model.add(Dense(512, activation="relu"))

    # Output layer (from 0 to 6 if 7 expressions)
    model.add(Dense(NBCLASSES, activation="softmax"))

    # Compilate cnn
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# IMSIZE = [100, 100]
IMSIZE = [48, 48]
# IMSIZE = [224, 224]

NBCLASSES = 7

batch_size = 32

epochs = 100

src_path_train = "../../../Databases/Fruits/fruits-360_dataset/fruits-360/Training"
src_path_train = "../../../Databases/FER-2013/train"
# src_path_train = "../../Databases/AffectNet/train_class"
src_path_test = "../../../Databases/Fruits/fruits-360_dataset/fruits-360/Test"
src_path_test = "../../../Databases/FER-2013/test"
# src_path_test = "../../Databases/AffectNet/val_class"

image_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    # rotation_range=20,
    # zoom_range=0.05,
    # width_shift_range=0.05,
    # height_shift_range=0.05,
    # shear_range=0.05,
    # horizontal_flip=True,
    fill_mode="nearest",
    # validation_split=0.20,
)


# create generators
train_generator = image_gen.flow_from_directory(
    src_path_train,
    target_size=IMSIZE,
    shuffle=True,
    batch_size=batch_size,
    # color_mode="grayscale"
)

test_generator = image_gen.flow_from_directory(
    src_path_test,
    target_size=IMSIZE,
    shuffle=True,
    batch_size=batch_size,
    # color_mode="grayscale"
)

from glob import glob

train_image_files = glob(src_path_train + "/*/*.jp*g")
test_image_files = glob(src_path_test + "/*/*.jp*g")
# len(image_files), len(valid_image_files)

mymodel = create_model()


early_stop = EarlyStopping(monitor="val_loss", patience=2)
# r = mymodel.fit_generator(
#     train_generator,
#     validation_data=test_generator,
#     epochs=epochs,
#     steps_per_epoch=len(train_image_files) // batch_size,
#     validation_steps=len(test_image_files) // batch_size,
#     callbacks=[early_stop],
# )
r = mymodel.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    steps_per_epoch=len(train_image_files) // batch_size,
    validation_steps=len(test_image_files) // batch_size,
    callbacks=[early_stop],
)

score = mymodel.evaluate(test_generator)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
