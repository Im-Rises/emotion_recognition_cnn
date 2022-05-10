import csv
import os

import numpy
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


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


def load_dataset(path):
    X = []
    y = []
    dummy_value = 0
    for folder in os.listdir(path):
        path_folder = path + folder + "/"
        for image in os.listdir(path_folder):
            X.append(np.array(Image.open(path_folder + image).getdata()))  # read image
            y.append(dummy_value)  # get dummy value
        dummy_value += 1
    X = np.array(X)
    y = np.asarray(y)
    return X, y


# Only implemented for 1 layer image (Grayscale images)
def preprocess_images(X, y, images_shape):
    # dimensions parameter needs to be a tuple with (width, height, number of layer)
    X = X / 255.0
    X = X.reshape(-1, images_shape[0], images_shape[1], 1)
    return X, y


def show_image(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.gray()
    plt.show()


def create_csv_from_dataset(filename, header, path):
    X, y = load_dataset(path)
    f = open(filename, "w")
    f.write(header)
    for emotion, image in zip(y, X):
        f.write("\n" + str(emotion) + ";")
        for pixel in image:
            f.write(str(pixel) + " ")


def load_csv_dataset(filename, x_header, y_header):
    df = pd.read_csv(filename, sep=";")
    y = df[y_header]
    images_list = df[x_header]
    # images_list = df[x_header].to_numpy()
    # X = []
    # for image in images_list:
    #     X.append(np.fromstring(image, sep=" "))
    # return np.asarray(X), np.asarray(y)
