from keras.layers import Dropout

import DeepLearningCnn as DLCnn

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


class ZFNET(DLCnn.DeepLearningCnn):
    def _create_compile_model(self):
        self._model.add(
            Conv2D(96, (7, 7), strides=(2, 2), input_shape=(224, 224, 3), padding='valid', activation='relu',
                   kernel_initializer='uniform'))
        self._model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self._model.add(
            Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform'))
        self._model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self._model.add(
            Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self._model.add(
            Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self._model.add(
            Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self._model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        self._model.add(Flatten())
        self._model.add(Dense(4096, activation='relu'))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(4096, activation='relu'))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(1000, activation='softmax'))
        self._model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self._model.summary()
