import os

import pandas as pd
from keras.applications.resnet import ResNet50

from Models import handle_dataset as hdtst
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from glob import glob
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow


def create_model():
    vgg = VGG19(
        input_shape=IMSIZE + [3],
        # weights="imagenet",
        include_top=False,
    )

    # Freeze existing VGG already trained weights
    for layer in vgg.layers:
        layer.trainable = False

    # get the VGG output
    out = vgg.output

    # Add new dense layer at the end
    x = Flatten()(out)
    x = Dense(NBCLASSES, activation="softmax")(x)

    model = Model(inputs=vgg.input, outputs=x)

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.summary()

    return model


# IMSIZE = [100, 100]
IMSIZE = [48, 48]
# IMSIZE = [224, 224]

NBCLASSES = 7

batch_size = 32

epochs = 100

src_path_train = "../../Databases/FER-2013/train"
# src_path_train = "../../Databases/AffectNet/train_class"
src_path_test = "../../Databases/FER-2013/test"
# src_path_test = "../../Databases/AffectNet/val_class"

image_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    # rotation_range=20,
    # zoom_range=0.05,
    # width_shift_range=0.05,
    # height_shift_range=0.05,
    # shear_range=0.05,
    # horizontal_flip=True,
    # fill_mode="nearest",
    # validation_split=0.20,
)

# create generators
train_generator = image_gen.flow_from_directory(
    src_path_train,
    target_size=IMSIZE,
    shuffle=True,
    batch_size=batch_size,
    color_mode="rgb",
)

test_generator = image_gen.flow_from_directory(
    src_path_test,
    target_size=IMSIZE,
    shuffle=True,
    batch_size=batch_size,
)


train_image_files = glob(src_path_train + "/*/*.jp*g")
test_image_files = glob(src_path_test + "/*/*.jp*g")
# len(image_files), len(valid_image_files)

mymodel = create_model()

# early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
r = mymodel.fit_generator(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    steps_per_epoch=len(train_image_files) // batch_size,
    validation_steps=len(test_image_files) // batch_size,
    # callbacks=[early_stop],
)

score = mymodel.evaluate(test_generator)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

mymodel.save("vggSavedModel")
