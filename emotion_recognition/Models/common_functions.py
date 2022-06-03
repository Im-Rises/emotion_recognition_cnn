from glob import glob

from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense
from keras.models import save_model
from keras.optimizer_v2.gradient_descent import SGD
from keras_preprocessing.image import ImageDataGenerator


def get_data(parameters, preprocess_input: object) -> tuple:
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
        parameters["train_path"], target_size=parameters["shape"], shuffle=True, batch_size=parameters["batch_size"]
    )

    test_generator = image_gen.flow_from_directory(
        parameters["test_path"], target_size=parameters["shape"], shuffle=True, batch_size=parameters["batch_size"]
    )

    return (
        glob(f"{parameters['train_path']}/*/*.jp*g"),
        glob(f"{parameters['test_path']}/*/*.jp*g"),
        train_generator,
        test_generator,
    )


def create_model(
        architecture, parameters
):
    model = architecture(
        input_shape=parameters["shape"] + [3],
        weights="imagenet",
        include_top=False,
        classes=parameters["nbr_classes"],
    )

    # Freeze existing VGG already trained weights
    for layer in model.layers[:parameters["number_of_last_layers_trainable"]]:
        layer.trainable = False

    # get the VGG output
    out = model.output

    # Add new dense layer at the end
    x = Flatten()(out)
    x = Dense(parameters["nbr_classes"], activation="softmax")(x)

    model = Model(inputs=model.input, outputs=x)

    opti = SGD(lr=parameters["learning_rate"], momentum=parameters["momentum"], nesterov=parameters["nesterov"])

    model.compile(
        loss="categorical_crossentropy", optimizer=opti, metrics=["accuracy"]
    )

    # model.summary()

    return model


def fit(
        model,
        train_generator,
        test_generator,
        train_files,
        test_files,
        parameters
):
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
