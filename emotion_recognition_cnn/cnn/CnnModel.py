import DeepLearningCnn as DLCnn

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


class CnnModel(DLCnn.DeepLearningCnn):
    # def __init__(self, number_of_emotions, images_shape, saved_model_path):
    #     super().__init__(number_of_emotions, images_shape, saved_model_path)

    def _create_compile_model(self):
        # print(self._number_of_emotions)
        # print(self._images_shape)
        # 3 convolution layers with progressive filter 32, 64 and 128
        self._model.add(
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                input_shape=self._images_shape,
                activation="relu",
            )
        )
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                input_shape=self._images_shape,
                activation="relu",
            )
        )
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                input_shape=self._images_shape,
                activation="relu",
            )
        )
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten data
        self._model.add(Flatten())

        # ANN classic layer
        self._model.add(Dense(512, activation="relu"))

        # Output layer (from 0 to 6 if 7 expressions)
        self._model.add(Dense(self._number_of_emotions, activation="softmax"))

        # Compilate cnn
        self._model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self._model.summary()
