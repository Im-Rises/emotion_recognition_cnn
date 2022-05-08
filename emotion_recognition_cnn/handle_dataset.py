import os
import numpy as np
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
    return X, y


# Only implemented for 1 layer image (Grayscale images)
def preprocess_images(X, y, images_shape):
    # dimensions parameter needs to be a tuple with (width, height, number of layer)
    X = np.array(X)
    X = X / 255.0
    X = X.reshape(-1, images_shape[0], images_shape[1], 1)
    y = np.asarray(y)
    return X, y


def show_image(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.gray()
    plt.show()
