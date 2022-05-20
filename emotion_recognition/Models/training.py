from keras.applications.resnet import ResNet50
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception

from emotion_recognition.Models.common_functions import (
    create_model,
    get_data,
    fit,
    evaluation_model,
    saveModel,
)

if __name__ == "__main__":
    shape = [48, 48]
    nbr_classes = 7
    train_path = "../../Databases/FER-2013/train/"
    test_path = "../../Databases/FER-2013/test/"
    batch_size = 16
    epochs = 20
    model, filename = None, None

    choice = input(
        "which models do you want to train?"
        "\n\t-1- resnet"
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
            f"batch_size = {batch_size} "
            f"\nepochs = {epochs} "
            f"\nshape = {shape} "
            f"\nnbr_classes = {nbr_classes} "
            f"\ntrain_path = {train_path} "
            f"\ntest_path = {test_path}"
            "above parameters for the train, if you want another parameters, you can change them directly in "
            "training.py "
        )
        train_files, test_files, train_generator, test_generator = get_data(
            train_path=train_path,
            test_path=test_path,
            shape=shape,
            batch_size=batch_size,
            preprocess_input=preprocess_input,
        )

        model = create_model(
            architecture=model,
            shape=shape,
            nbr_classes=nbr_classes,
        )

        history = fit(
            model=model,
            train_generator=train_generator,
            test_generator=test_generator,
            epochs=epochs,
            train_files=train_files,
            test_files=test_files,
            batch_size=batch_size,
        )

        evaluation_model(model, test_generator)

        saveModel(filename)
