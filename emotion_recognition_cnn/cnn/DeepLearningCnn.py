# Abstract class
from abc import ABC, abstractmethod

# Tensorflow and keras
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import load_model

# Other libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# My libs
from emotion_recognition_cnn import handle_dataset as hdtst


class DeepLearningCnn(ABC):
    def __init__(self, number_of_emotions, images_shape, saved_model_path):
        self._number_of_emotions = number_of_emotions
        self._images_shape = images_shape
        self._model = tf.keras.Sequential()
        self._create_compile_model()
        self.__already_trained = False

    def __init__(self, number_of_emotions, images_shape, saved_model_path):
        self._number_of_emotions = number_of_emotions
        self._images_shape = images_shape
        self._model = tf.keras.Sequential()
        if saved_model_path:
            try:
                self.__load_model_architecture(saved_model_path)
                self.__already_trained = True
            except:
                self._create_compile_model()
                self.__already_trained = False
        else:
            self._create_compile_model()
            self.__already_trained = False

    @abstractmethod
    def _create_compile_model(self):
        pass

    def __load_model_architecture(self, path):
        self._model = load_model(path)
        # self._model = self._model.load_weights(path)
        print("Model loaded")
        self._model.summary()

    def fit(self, X_train, y_train, X_test, y_test):
        if self.__already_trained:
            print("Model loaded and already trained")
        else:
            print("Model not trained, training model")
            early_stop = EarlyStopping(monitor="val_loss", patience=2)
            self._model.fit(
                x=X_train,
                y=y_train,
                validation_data=(X_test, y_test),
                epochs=25,
                shuffle=True,
                callbacks=[early_stop],
            )

            self.plot_accuracies()

    def plot_accuracies(self):
        # Print accuracy and losses
        losses = pd.DataFrame(self._model.history.history)
        losses[["accuracy", "val_accuracy"]].plot()
        losses[["loss", "val_loss"]].plot()
        plt.show()

    def predict_image(self, img, emotion_value, dic):
        predicted_emotion_value = np.argmax(
            self._model.predict(img.reshape(1, 48, 48, 1)), axis=-1
        )[0]
        hdtst.show_image(
            img,
            "Real : {}, Predicted : {}".format(
                dic[emotion_value], dic[predicted_emotion_value]
            ),
        )

    def predict_images(self):
        print("To implement predict_images")
        pass

    def save(self, name):
        self._model.save(name)
        # self._model.save_weights(name)
