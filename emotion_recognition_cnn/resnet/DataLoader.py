import os

import cv2
import numpy as np
from keras.backend import expand_dims
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class DataLoader:
    def __init__(
        self,
        path: str,
        labels: list,
        labels_limit: list = None,
        shape: tuple = (48, 48, 3),
        max_label: int = 50,
    ):
        self._labels_limit = labels_limit if labels_limit is not None else []
        self._path = path
        self._data_dir_list = os.listdir(path)
        self._img_data_list = []
        self._shape = shape
        self._labels_name = labels
        self._labels = []
        self._max_label = max_label

        self._load_data()

    def _load_data(self) -> None:
        for dataset in self._data_dir_list:
            img_list = os.listdir(self._path + "/" + dataset)[0 : self._max_label]
            print("Loaded the images of dataset-" + "{}\n".format(dataset))
            nbr_img_of_this_label = 0
            for img in img_list:
                input_img = cv2.imread(self._path + "/" + dataset + "/" + img)
                # input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                # input_img = input_img.reshape(32, 32, 3)
                input_img = np.array(cv2.resize(input_img, (32, 32)))
                # print(input_img.shape)
                input_img = expand_dims(input_img, axis=0)
                self._img_data_list.append(input_img)

        self._img_data_list = np.array(self._img_data_list).astype(np.float32) / 255

        if self._labels_limit is not None:
            self.set_labels()

    def train_test_data(self) -> tuple:
        y = np_utils.to_categorical(self._labels, len(self._labels_name))

        x, y = shuffle(self._img_data_list, y, random_state=2)
        # return (x_train, x_test, y_train, y_test)
        return train_test_split(x, y, test_size=0.15, random_state=8)

    def set_labels(self) -> None:
        nbr_samples = self._img_data_list.shape[0]
        self._labels = np.ones((nbr_samples,), dtype=np.int64)
        acc, value = 0, 0
        for id in self._labels_limit:
            self._labels[acc:id] = value
            acc = id + 1
            value += 1

    def get_label(self, id: int) -> str:
        return self._labels_name[id]
