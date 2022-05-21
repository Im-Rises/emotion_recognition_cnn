import os

from keras.applications.resnet import ResNet50
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception
from matplotlib import pyplot as plt

from common_functions import (
    create_model,
    get_data,
    fit,
    evaluation_model,
    saveModel,
)

if __name__ == "__main__":
    parameters = {
        "shape": [80, 80],
        "nbr_classes": 7,
        "train_path": "../../Databases/FER-2013/train/",
        "test_path": "../../Databases/FER-2013/test/",
        "batch_size": 8,
        "epochs": 30,
        "number_of_last_layers_trainable": 5,
    }
    model, filename = None, None

    choice = input(
        "which models do you want to train?"
        "\n\t-1- resnet50"
        "\n\t-2- vgg19"
        "\n\t-3- xception"
        "\n>>>"
    )
    if choice == "1":
        model = ResNet50
        filename = "resnet50"
    elif choice == "2":
        model = VGG19
        filename = "vgg19"
    elif choice == "3":
        model = Xception
        filename = "xception"
    else:
        print("you have to choose a number between 1 and 3")
        exit(1)

    if model is not None and filename is not None:
        print(
            f"{parameters}above parameters for the train, if you want another parameters, you can change them "
            f"directly in training.py "
        )

        train_files, test_files, train_generator, test_generator = get_data(
            train_path=parameters["train_path"],
            test_path=parameters["test_path"],
            shape=parameters["shape"],
            batch_size=parameters["batch_size"],
            preprocess_input=preprocess_input,
        )

        model = create_model(
            architecture=model,
            shape=parameters["shape"],
            nbr_classes=parameters["nbr_classes"],
            number_of_last_layers_trainable=5,
        )

        history = fit(
            model=model,
            train_generator=train_generator,
            test_generator=test_generator,
            epochs=parameters["epochs"],
            train_files=train_files,
            test_files=test_files,
            batch_size=parameters["batch_size"],
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
