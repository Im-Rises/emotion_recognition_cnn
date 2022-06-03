import os
from glob import glob

from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import (
    Flatten,
    Dense,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Activation,
    Dropout,
)
from keras.models import save_model, Sequential
from keras.optimizer_v2.gradient_descent import SGD
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow import keras


def get_data(parameters, preprocess_input=None) -> tuple:
    image_gen = ImageDataGenerator(
        # rescale=1 / 127.5,
        rotation_range=20,
        zoom_range=0.05,
        shear_range=10,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20,
        preprocessing_function=preprocess_input,
    )

    # create generators
    train_generator = image_gen.flow_from_directory(
        parameters["train_path"],
        target_size=parameters["shape"],
        shuffle=True,
        batch_size=parameters["batch_size"],
    )

    test_generator = image_gen.flow_from_directory(
        parameters["test_path"],
        target_size=parameters["shape"],
        shuffle=True,
        batch_size=parameters["batch_size"],
    )

    return (
        glob(f"{parameters['train_path']}/*/*.jp*g"),
        glob(f"{parameters['test_path']}/*/*.jp*g"),
        train_generator,
        test_generator,
    )


def fine_tuning(model: Model, parameters):
    # fine tuning
    for layer in model.layers[: parameters["number_of_last_layers_trainable"]]:
        layer.trainable = False
    return model


def create_model(architecture, parameters):
    model = architecture(
        input_shape=parameters["shape"] + [3],
        weights="imagenet",
        include_top=False,
        classes=parameters["nbr_classes"],
    )

    # get the model output
    out = model.output

    # Add new dense layer at the end
    x = Flatten()(out)
    x = Dense(parameters["nbr_classes"], activation="softmax")(x)

    model = Model(inputs=model.input, outputs=x)

    opti = SGD(
        lr=parameters["learning_rate"],
        momentum=parameters["momentum"],
        nesterov=parameters["nesterov"],
    )

    model.compile(loss="categorical_crossentropy", optimizer=opti, metrics=["accuracy"])

    return model


def alexnet_model(parameters):
    model = Sequential()
    model.add(
        Conv2D(
            96,
            (11, 11),
            input_shape=parameters["shape"] + [3],
            activation="relu",
            padding="same",
        )
    )
    model.add(Conv2D(256, (5, 5), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(384, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(384, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(384, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation("softmax"))  # Formats data to make them be like probabilities

    model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
    )

    return model


def fit(model, train_generator, test_generator, train_files, test_files, parameters):
    early_stop = EarlyStopping(monitor="val_accuracy", patience=2)
    return model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=parameters["epochs"],
        steps_per_epoch=len(train_files) // parameters["batch_size"],
        validation_steps=len(test_files) // parameters["batch_size"],
        callbacks=[early_stop],
    )


def evaluation_model(model, test_generator):
    score = model.evaluate_generator(test_generator)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return score


def saveModel(filename, model):
    save_model(model=model, filepath=f"./trained_models/{filename}")


if __name__ == "__main__":
    preprocess_input = keras.applications.resnet.preprocess_input
    filename = "alexnet"

    parameters = {
        "shape": [80, 80],
        "nbr_classes": 7,
        "train_path": "../../Databases/FER-2013/train/",
        "test_path": "../../Databases/FER-2013/test/",
        "batch_size": 8,
        "epochs": 50,
        "number_of_last_layers_trainable": 10,
        "learning_rate": 0.001,
        "nesterov": True,
        "momentum": 0.9,
    }
    train_files, test_files, train_generator, test_generator = get_data(
        preprocess_input=preprocess_input, parameters=parameters
    )

    model = alexnet_model(parameters)

    history = fit(
        model=model,
        train_generator=train_generator,
        test_generator=test_generator,
        train_files=train_files,
        test_files=test_files,
        parameters=parameters,
    )

    score = evaluation_model(model, test_generator)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0, 1.0])
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()

    if os.path.isfile(f"./trained_models/{filename}_parameters.txt"):
        with open(f"./trained_models/{filename}_parameters.txt", "r") as file:
            print(file.read())
            file.close()

    choice = input("save model? (O/N)\n>>>")

    if choice == "O":
        saveModel(filename=filename, model=model)
        with open(f"./trained_models/{filename}_parameters.txt", "w") as file:
            file.write(f"{parameters}\nval_acc: {val_acc}\nval_loss: {val_loss}")
            file.close()
