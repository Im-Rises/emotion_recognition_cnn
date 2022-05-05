import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
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
    test_dir = '../Databases/FER-2013/train/'
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
    # rotation_range=40,
    #         width_shift_range=0.2,
    #         height_shift_range=0.2,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         horizontal_flip=True,
    #         fill_mode='nearest'

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

    cnn = tf.keras.Sequential()
    cnn.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(48, 48, 1), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(48, 48, 1), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(tf.keras.layers.Conv2D(24, (3, 3), activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(24, (3, 3), activation='relu'))
    cnn.add(tf.keras.layers.Conv2D(12, (3, 3), activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling2D(3, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(4096))
    cnn.add(tf.keras.layers.Dense(4096))
    cnn.add(tf.keras.layers.Dense(7, activation='softmax'))
    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    cnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    cnn.summary()

    steps_per_epoch = train_set.n
    validation_steps = test_set.n

    cnn.fit(x=train_set,
            validation_data=test_set,
            epochs=60,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[early_stop]
            )
