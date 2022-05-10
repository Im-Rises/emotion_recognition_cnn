import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools

from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
import keras.optimizers
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers import BatchNormalization

np.random.seed(2)


def preprocess(X):
    X = np.array([np.fromstring(image, np.uint8, sep=" ") for image in X])
    X = X / 255.0
    X = X.reshape(-1, 48, 48, 1)
    return X


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    # Predict the values from the validation dataset
    Y_pred = model.predict(X_val)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_val, axis=1)
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes=range(10))


# Source : https://github.com/pranjalrai-iitd/FER2013-Facial-Emotion-Recognition-/blob/master/emotion.ipynb


if __name__ == "__main__":
    data = pd.read_csv("fer2013.csv")
    print(data.head())
    groups = [g for _, g in data.groupby("Usage")]
    train = groups[2]
    val = groups[1]
    test = groups[0]
    train = train.drop(labels=["Usage"], axis=1)
    val = val.drop(labels=["Usage"], axis=1)
    test = test.drop(labels=["Usage"], axis=1)

    Y_train = train["emotion"]
    Y_val = val["emotion"]
    Y_test = test["emotion"]
    # 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'#

    X_train = train["pixels"]
    X_val = val["pixels"]
    X_test = test["pixels"]

    X_train = preprocess(X_train)
    X_val = preprocess(X_val)
    X_test = preprocess(X_test)
    plt.imshow(X_train[0][:, :, 0])

    plt.figure(figsize=(30, 7))

    plt.subplot(1, 3, 1)
    ax = sns.countplot(Y_train)
    ax.set(ylabel="count", xlabel="emotion")
    plt.title("Counts per emotion in training set")

    plt.subplot(1, 3, 2)
    ax = sns.countplot(Y_val)
    ax.set(ylabel="count", xlabel="emotion")
    plt.title("Counts per emotion in validation set")

    plt.subplot(1, 3, 3)
    ax = sns.countplot(Y_test)
    ax.set(ylabel="count", xlabel="emotion")
    plt.title("Counts per emotion in testing set")

    print("Is any label null in training set:", Y_train.isnull().any())
    print("Is any label null in validation set:", Y_val.isnull().any())
    print("Is any label null in testing set:", Y_test.isnull().any())

    Y_train = to_categorical(Y_train, num_classes=7)
    Y_val = to_categorical(Y_val, num_classes=7)
    Y_test = to_categorical(Y_test, num_classes=7)

    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), padding="Same", activation="relu", input_shape=(48, 48, 1))
    )
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), padding="Same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), padding="Same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding="Same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding="Same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (5, 5), padding="Same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(7, activation="softmax"))

    optimizer = keras.optimizers.adam_v2.Adam(lr=0.001)
    lr_anneal = ReduceLROnPlateau(
        monitor="val_accuracy", patience=3, factor=0.2, min_lr=1e-6
    )
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=50,
        batch_size=100,
        callbacks=[lr_anneal],
    )

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history["loss"], color="b", label="Training loss")
    ax[0].plot(
        history.history["val_loss"], color="r", label="validation loss", axes=ax[0]
    )
    legend = ax[0].legend(loc="best", shadow=True)

    ax[1].plot(history.history["accuracy"], color="b", label="Training accuracy")
    ax[1].plot(history.history["val_accuracy"], color="r", label="Validation accuracy")
    legend = ax[1].legend(loc="best", shadow=True)

    score, acc = model.evaluate(X_test, Y_test, batch_size=100)
    print("Test score:", score)
    print("Test accuracy:", acc)
