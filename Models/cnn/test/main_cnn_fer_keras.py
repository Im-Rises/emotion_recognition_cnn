import numpy as np
from keras.callbacks import EarlyStopping
from glob import glob

from keras.engine.input_layer import InputLayer
from keras_preprocessing.image import ImageDataGenerator

from Models.cnn.cnn_model import CnnModel
from Models import handle_dataset as hdtst
from random import randrange
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf

if __name__ == "__main__":
    # Fer-13 variables
    train_path = "../../../Databases/FER-2013/train/"
    test_path = "../../../Databases/FER-2013/test/"
    train_csv_file = "train_fer.csv"
    test_csv_file = "test_fer.csv"
    saved_model_name = "cnn_fer_model"
    number_of_emotions = 7
    images_shape = [48, 48]
    batch_size = 2

    # Create dictionaries
    value_emotion_dic = hdtst.create_dictionary(train_path)

    # # Load dataset
    # X_train, y_train = hdtst.load_dataset(train_path, images_shape)
    # X_test, y_test = hdtst.load_dataset(test_path, images_shape)
    #
    # # Preprocess images and image label
    # X_train = hdtst.preprocess_images(X_train, images_shape)
    # X_test = hdtst.preprocess_images(X_test, images_shape)

    image_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
    )

    train_generator = image_gen.flow_from_directory(
        train_path,
        target_size=images_shape,
        shuffle=True,
        batch_size=batch_size,
        color_mode="grayscale",
    )

    test_generator = image_gen.flow_from_directory(
        test_path,
        target_size=images_shape,
        shuffle=True,
        batch_size=batch_size,
        color_mode="grayscale",
    )

    # Create CNN
    cnn = tf.keras.Sequential()

    cnn.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            # input_shape=images_shape+[1],
            activation="relu",
        )
    )
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            # input_shape=images_shape+[1],
            activation="relu",
        )
    )
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            # input_shape=images_shape+[1],
            activation="relu",
        )
    )
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten data
    cnn.add(Flatten())

    # ANN classic layer
    cnn.add(Dense(512, activation="relu"))

    # Output layer (from 0 to 6 if 7 expressions)
    cnn.add(Dense(number_of_emotions, activation="softmax"))

    # Compilate cnn
    cnn.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cnn.summary()

    # Fit and predict
    early_stop = EarlyStopping(monitor="val_loss", patience=2)
    # cnn.fit(
    #     x=X_train,
    #     y=y_train,
    #     validation_data=(X_test, y_test),
    #     epochs=25,
    #     shuffle=True,
    #     callbacks=[early_stop],
    # )

    train_image_files = glob(train_path + "/*/*.jp*g")
    test_image_files = glob(test_path + "/*/*.jp*g")

    print(len(train_image_files) // batch_size)
    print(len(test_image_files) // batch_size)
    cnn.fit(
        x=train_generator,
        validation_data=test_generator,
        epochs=25,
        steps_per_epoch=len(train_image_files) // batch_size,
        validation_steps=len(test_image_files) // batch_size,
        callbacks=[early_stop],
    )

    # cnn.save(saved_model_name)

    # # Predict one image
    # random_image_index = randrange(len(X_test))
    # predicted_emotion_value = np.argmax(
    #     cnn.predict(X_test[random_image_index].reshape(1, 48, 48, 1)), axis=-1
    # )[0]
    # hdtst.show_image(
    #     X_test[random_image_index],
    #     "Real : {}, Predicted : {}".format(value_emotion_dic[y_test[random_image_index]],
    #                                        value_emotion_dic[predicted_emotion_value]),
    # )
    #
    #
    # my_X_test, my_y_test = hdtst.load_dataset("../../Databases/My_test/", images_shape)
    # my_X_test = hdtst.preprocess_images(my_X_test, images_shape)
    # for image, emotion in zip(my_X_test, my_y_test):
    #     predicted_emotion_value = np.argmax(
    #         cnn.predict(image.reshape(1, 48, 48, 1)), axis=-1
    #     )[0]
    #     hdtst.show_image(
    #         image,
    #         "Real : {}, Predicted : {}".format(
    #             value_emotion_dic[emotion], value_emotion_dic[predicted_emotion_value]
    #         ),
    #     )
