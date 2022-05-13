import csv
import os

import numpy
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img


def create_dictionary(path):
    dic = {}
    value = 0
    for emotion in os.listdir(path):
        dic[value] = emotion
        value += 1
    # dic_reversed = {v: e for e, v in dic.items()}  # Create dictionary in the other sens
    print("Keys and expression equivalents :")
    for value, emotion in dic.items():
        print("  {} <=> {}".format(value, emotion))
    return dic


def load_dataset_v2(path, target_shape):
    X = []
    y = []
    dummy_value = 0
    for folder in os.listdir(path):
        path_folder = path + folder + "/"
        for image in os.listdir(path_folder):
            X.append(
                np.array(
                    load_img(
                        path_folder + image,
                        grayscale=True,
                        target_size=target_shape,
                    ).getdata()
                )
            )
            y.append(dummy_value)  # get dummy value
            dummy_value += 1
    X = np.array(X)
    y = np.array(y)
    return X, y


# Only implemented for 1 layer image (Grayscale images)
def preprocess_images(X, images_shape):  # -1 means unknown size
    # dimensions parameter needs to be a tuple with (width, height, number of layer)
    X = X / 255.0
    X = X.reshape(-1, images_shape[0], images_shape[1], images_shape[2])
    return X


def show_image(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.gray()
    plt.show()
