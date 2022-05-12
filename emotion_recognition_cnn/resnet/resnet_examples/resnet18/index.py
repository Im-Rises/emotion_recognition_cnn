import os

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator

from emotion_recognition_cnn.resnet.DataLoader import DataLoader
from emotion_recognition_cnn.resnet.Resnet18 import ResNet18
from keras.callbacks import EarlyStopping


def plot_model_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(history.history["accuracy"])
    axs[0].plot(history.history["val_accuracy"])
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")

    axs[0].legend(["train", "validate"], loc="upper left")
    # summarize history for loss
    axs[1].plot(history.history["loss"])
    axs[1].plot(history.history["val_loss"])
    axs[1].set_title("Model Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(["train", "validate"], loc="upper left")
    plt.show()


"""
####### PREPROCESSING ######## 
"""

train_path = "./../../../../Databases/FER-2013/train/"
test_path = "./../../../../Databases/FER-2013/test/"

fer2013 = DataLoader(
    train_path=train_path,
    test_path=test_path,
    labels=os.listdir(train_path),
    # max_img_per_folder=2,
    shape=(32, 32, 3),
)

x_train, y_train = fer2013.get_train_data()
x_test, y_test = fer2013.get_test_data()

X_train = x_train.astype(np.float)
X_test = x_test.astype(np.float)
X_train /= 255.0
X_test /= 255.0

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=True
)

# encoder = OneHotEncoder()
# encoder.fit(y_train)

# y_train = encoder.transform(y_train).toarray()
# y_test = encoder.transform(y_test).toarray()
# y_val = encoder.transform(y_val).toarray()

"""
####### DATA AUGMENTATION #######
"""

aug = ImageDataGenerator(
    horizontal_flip=True, width_shift_range=0.05, height_shift_range=0.05
)
# aug.fit(X_train)

print(X_train.shape)
"""
####### resnet18 model #######
"""
try:
    model = load_model("savedModel")
    model.summary()
except:
    model = ResNet18(7)
    model.build(input_shape=(None, 32, 32, 3))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    es = EarlyStopping(patience=8, restore_best_weights=True, monitor="val_acc")
    # I did not use cross validation, so the validate performance is not accurate.
    STEPS = len(X_train) / 256

    train = (X_train, y_train)
    history = model.fit(
        x=np.stack(X_train),
        y=np.stack(y_train),
        steps_per_epoch=STEPS,
        batch_size=256,
        epochs=50,
        validation_data=(X_test, y_test),
        # callbacks=[es],
    )

    model.save("savedModel")

"""
####### TRAIN CURVE #######
"""

"""
# list all data in history
print(history.history.keys())
plotmodelhistory(history)
"""

"""
####### PREDICTION #######
"""
## Evaluation

ModelLoss, ModelAccuracy = model.evaluate(X_test, y_test)

print("Model Loss is {}".format(ModelLoss))
print("Model Accuracy is {}".format(ModelAccuracy))

print(type(model.model))
