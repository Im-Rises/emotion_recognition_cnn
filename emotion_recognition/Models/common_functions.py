from glob import glob

from keras import Model
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense
from keras.models import save_model
from keras_preprocessing.image import ImageDataGenerator


def get_data(
        train_path: str,
        test_path: str,
        shape: list,
        batch_size: int,
        preprocess_input: object,
) -> tuple:
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
        train_path, target_size=shape, shuffle=True, batch_size=batch_size
    )

    test_generator = image_gen.flow_from_directory(
        test_path, target_size=shape, shuffle=True, batch_size=batch_size
    )

    return (
        glob(f"{train_path}/*/*.jp*g"),
        glob(f"{test_path}/*/*.jp*g"),
        train_generator,
        test_generator,
    )


def create_model(architecture, shape: list, nbr_classes: int, number_of_last_layers_trainable: int):
    model = architecture(
        input_shape=shape + [3], weights="imagenet", include_top=False, classes=7
    )

    # Freeze existing VGG already trained weights
    for layer in model.layers[:number_of_last_layers_trainable]:
        layer.trainable = False

    # get the VGG output
    out = model.output

    # Add new dense layer at the end
    x = Flatten()(out)
    x = Dense(nbr_classes, activation="softmax")(x)

    model = Model(inputs=model.input, outputs=x)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # model.summary()

    return model


def fit(
        model,
        train_generator,
        test_generator,
        epochs,
        train_files,
        test_files,
        batch_size,
):
    early_stop = EarlyStopping(monitor="val_accuracy", patience=2)
    return model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=epochs,
        steps_per_epoch=len(train_files) // batch_size,
        validation_steps=len(test_files) // batch_size,
        callbacks=[early_stop],
    )


def evaluation_model(model, test_generator):
    score = model.evaluate_generator(test_generator)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return score


def saveModel(filename, model):
    save_model(model=model, filepath=f"./trained_models/{filename}")
