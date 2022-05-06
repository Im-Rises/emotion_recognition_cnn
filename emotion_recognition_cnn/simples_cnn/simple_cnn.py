import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

if __name__ == '__main__':
    dataset_fashion_mnsit = tf.keras.datasets.fashion_mnist

    # Load train and test data
    (X_train, y_train), (X_test, y_test) = dataset_fashion_mnsit.load_data()
    print(X_train[0])
    # Train data repartition
    print(pd.DataFrame(y_train)[0].value_counts())

    # Data normalization
    X_train = X_train / 255
    X_test = X_test / 255

    # Train and test data shape
    print("Données entrainement: {}, Test: {}".format(X_train.shape, X_test.shape))

    # # Plot an image from the dataset test
    # plt.imshow(X_train[0])
    # plt.title("Tag {}".format(y_train[0]))
    # plt.show()

    # We need an image in grayscale, for this purpose we delete the 2 of the layer of the images (RGB => Grayscale).
    # Resizing images function (from numpy)
    # 60000 number of images
    # 28x28 image' dimension (so it's not chaning it)
    # 1 grayscale (3 for RGB)
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    try:
        cnn = load_model("savedModel")
        img = X_train[0]
        print("Image 0 prediction : {}".format(np.argmax(cnn.predict(img.reshape(1, 28, 28, 1)), axis=-1)[0]))
    except:
        # CNN
        cnn = tf.keras.Sequential()

        # 3 convolution layers with progressive filter 32, 64 and 128
        cnn.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))

        cnn.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))

        cnn.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten data
        cnn.add(Flatten())

        # ANN classic layer
        cnn.add(Dense(512, activation='relu'))

        # Output layer (from 0 to 9)
        cnn.add(Dense(10, activation='softmax'))

        # earlyStopping to know the number of epoch to do.
        early_stop = EarlyStopping(monitor='val_loss', patience=2)

        # Compilate CNN
        cnn.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        cnn.summary()

        plt.imshow(X_train[0])
        plt.gray()
        plt.show()
        print((X_train[0][14]))
        print(type(X_train[0]))

        # Training
        cnn.fit(x=X_train,
                y=y_train,
                validation_data=(X_test, y_test),
                epochs=25,
                callbacks=[early_stop])

        # Print accuracy and losses
        losses = pd.DataFrame(cnn.history.history)
        losses[['accuracy', 'val_accuracy']].plot()

        losses[['loss', 'val_loss']].plot()

        # Predict test
        img = X_train[0]
        print("Image 0 prediction : {}".format(np.argmax(cnn.predict(img.reshape(1, 28, 28, 1)), axis=-1)[0]))

        # Save last model
        # cnn.save("savedModel")
