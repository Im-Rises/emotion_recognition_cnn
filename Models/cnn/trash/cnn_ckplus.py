import os

import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split


def create_dictionary(path):
    dic = {}
    value = 0
    for emotion in os.listdir(path):
        dic[value] = emotion
        value += 1

    dic_reversed = {v: e for e, v in dic.items()}  # Create dictionary in the other sens

    print("Keys and expression equivalents :")
    for value, emotion in dic.items():
        print("  {} <=> {}".format(value, emotion))

    return dic, dic_reversed


def load_dataset(path, dic):
    X = []
    y = []
    for folder in os.listdir(path):
        path_folder = path + folder + "/"
        for image in os.listdir(path_folder):
            X.append(np.array(Image.open(path_folder + image).getdata()))  # read image
            y.append(dic[folder])  # get dummy value
    return X, y


def preprocess_images(X, y):
    X = np.array(X)
    X = X / 255.0
    X = X.reshape(-1, 48, 48, 1)
    y = np.asarray(y)
    return X, y


def show_image(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.gray()
    plt.show()


def predict_image(img, emotion_value, dic):
    predicted_emotion_value = np.argmax(
        cnn.predict(img.reshape(1, 48, 48, 1)), axis=-1
    )[0]
    show_image(
        img,
        "Real : {}, Predicted : {}".format(
            dic[emotion_value], dic[predicted_emotion_value]
        ),
    )


if __name__ == "__main__":
    saved_model_name = "cnn_ckplus_model"
    number_of_emotion = 7
    test_size = 0.2
    image_shape = (48, 48, 1)
    images_path = "../../Databases/CKPLUS/CK+48/"

    # Create dictionaries
    value_emotion_dic, emotion_value_dic = create_dictionary(images_path)

    # Load data
    X, y = load_dataset(images_path, emotion_value_dic)

    # Preprocess images and image label
    X, y = preprocess_images(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    try:
        cnn = load_model(saved_model_name)
        cnn.summary()
        predict_image(X_test[1], y_test[1], value_emotion_dic)

    except:
        # CNN
        cnn = tf.keras.Sequential()

        # 3 convolution layers with progressive filter 32, 64 and 128
        cnn.add(
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                input_shape=image_shape,
                activation="relu",
            )
        )
        cnn.add(MaxPooling2D(pool_size=(2, 2)))

        cnn.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                input_shape=image_shape,
                activation="relu",
            )
        )
        cnn.add(MaxPooling2D(pool_size=(2, 2)))

        cnn.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                input_shape=image_shape,
                activation="relu",
            )
        )
        cnn.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten data
        cnn.add(Flatten())

        # ANN classic layer
        cnn.add(Dense(512, activation="relu"))

        # Output layer (from 0 to 6 if 7 expressions)
        cnn.add(Dense(number_of_emotion, activation="softmax"))

        # earlyStopping to know the number of epoch to do.
        early_stop = EarlyStopping(monitor="val_loss", patience=2)

        # Compilate CNN
        cnn.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        cnn.summary()

        cnn.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_test, y_test),
            epochs=25,
            shuffle=True,
            callbacks=[early_stop],
        )

        # Print accuracy and losses
        losses = pd.DataFrame(cnn.history.history)
        losses[["accuracy", "val_accuracy"]].plot()
        losses[["loss", "val_loss"]].plot()
        plt.show()

        predict_image(X_test[0], y_test[0], value_emotion_dic)

        # Save last model
        cnn.save(saved_model_name)
        print("Model saved")
