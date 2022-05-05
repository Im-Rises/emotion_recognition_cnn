import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os


def count_expression(path, set_):
    dict_ = {}
    for expression in os.listdir(path):
        dir_ = path + expression
        dict_[expression] = len(os.listdir(dir_))

    df = pd.DataFrame(dict_, index=[set_])
    return df


if __name__ == '__main__':
    train_dir = '../Databases/FER-2013/train/'
    test_dir = '../Databases/FER-2013/test/'

    train_count = count_expression(train_dir, 'train_count')
    print(train_count)
    test_count = count_expression(test_dir, 'test')
    print(test_count)

    train_count.transpose().plot(kind="bar")
    plt.title('Plot of number of images in train dataset')
    plt.show()
    test_count.transpose().plot(kind="bar")
    plt.title('Plot of number of images in test dataset')
    plt.show()
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       zoom_range=0.3,
                                       horizontal_flip=True)

    train_set = train_datagen.flow_from_directory(train_dir,
                                                  batch_size=64,
                                                  target_size=(48, 48),
                                                  shuffle=True,
                                                  color_mode="grayscale", class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory(test_dir,
                                                batch_size=64,
                                                target_size=(48, 48),
                                                shuffle=True,
                                                color_mode="grayscale", class_mode='categorical')
    print(train_set.class_indices)

    steps_per_epoch = train_set.n
    validation_steps = test_set.n



    # CNN
    cnn = tf.keras.Sequential()

    # 3 convolution layers with progressive filter 32, 64 and 128
    cnn.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(48, 48, 1), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(48, 48, 1), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(48, 48, 1), activation='relu'))
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

    cnn.fit(x=train_set,
            epochs=60,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[early_stop]
            )