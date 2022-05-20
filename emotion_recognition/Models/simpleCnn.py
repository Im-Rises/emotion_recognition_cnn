import tensorflow as tf
import keras

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.layers import Input, Lambda, Dense, Flatten, Rescaling
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# def plot_image(i, predictions_array, true_label, img):
#     true_label, img = true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.imshow(img, cmap=plt.cm.binary)
#
#     predicted_label = np.argmax(predictions_array)
#     if predicted_label == true_label:
#         color = 'blue'
#     else:
#         color = 'red'
#
#     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                          100 * np.max(predictions_array),
#                                          class_names[true_label]),
#                color=color)
#
#
# def plot_value_array(i, predictions_array, true_label):
#     true_label = true_label[i]
#     plt.grid(False)
#     plt.xticks(range(10))
#     plt.yticks([])
#     thisplot = plt.bar(range(10), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)
#
#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label].set_color('blue')
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

if __name__ == "__main__":
    batch_size = 32
    img_height = 48
    img_width = 48

    train_path = "../../Databases/FER-2013/train"
    test_path = "../../Databases/FER-2013/test"

    num_classes = 7

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        # validation_split=0.2,
        # subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        # validation_split=0.2,
        # subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale",
    )

    class_names = train_ds.class_names
    print(class_names)

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

    # Tensors' and images' dimensions
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    # # NOT WORKING ?!?
    # # Images normalization
    # normalization_layer = tf.keras.layers.Rescaling(1. / 255) # [0,1]
    # # normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1) # [-1,-1]
    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_ds))
    # first_image = image_batch[0]
    # # Notice the pixel values are now in `[0,1]`.
    # print(np.min(first_image), np.max(first_image))

    # vgg = VGG16(input_shape=[img_height, img_width] + [3], weights='imagenet', include_top=False)
    # for layer in vgg.layers:
    #     layer.trainable = False
    #
    # # get the VGG output
    # out = vgg.output
    #
    # # Add new dense layer at the end
    # x = Flatten()(out)
    # x = Dense(num_classes, activation='softmax')(x)
    #
    # model = Model(inputs=vgg.input, outputs=x)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=2)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[early_stop],
    )

    # Show accuracy and val_accuracy
    losses = pd.DataFrame(model.history.history)
    losses[["accuracy", "val_accuracy"]].plot()
    plt.show()

    model.save("modelSimpleCnn")

    ## display some predictions
    # num_rows = 5
    # num_cols = 3
    # num_images = num_rows * num_cols
    # plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    # for i in range(num_images):
    #     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    #     plot_image(i, predictions[i], test_labels, test_images)
    #     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    #     plot_value_array(i, predictions[i], test_labels)
    # plt.tight_layout()
    # plt.show()

    ## Save model
    # model.save("model_saved")
