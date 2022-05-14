import os

import cv2
import numpy as np
from numpy import expand_dims


class DataLoader:
    def __init__(self, shape=None, **kargs):
        self.__labels_limit_test = None
        self.__max_img_per_folder = None
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None
        self.__labels_test = None
        self.__labels_train = None
        self.__labels_name = None
        self.__shape = shape
        self.__test_dir = None
        self.__train_dir = None
        self.__test_path = None
        self.__train_path = None
        self.__labels_limit_train = None

        if "train_path" in kargs:
            self.init1(
                train_path=kargs["train_path"],
                test_path=kargs["test_path"],
                labels=kargs["labels"],
                shape=self.__shape,
                max_img_per_folder=kargs["max_img_per_folder"]
                if "max_img_per_folder" in kargs
                else None,
            )
        else:
            self.init2(
                x_train=kargs["x_train"],
                y_train=kargs["y_train"],
                x_test=kargs["x_test"],
                y_test=kargs["x_test"],
            )

    def init1(
        self,
        train_path: str,
        test_path: str,
        labels: list,
        shape: tuple = (48, 48, 3),
        max_img_per_folder: int = None,
    ):
        self.__train_path = train_path
        self.__test_path = test_path
        self.__train_dir = os.listdir(train_path)
        self.__test_dir = os.listdir(test_path)

        self.x_train, self.y_train, self.x_test, self.y_test = [], [], [], []

        self.__labels_limit_train, self.__labels_limit_test = [], []
        self.__shape = shape
        self.__labels_name = labels
        self.__labels_train, self.__labels_test = [], []
        self.__max_img_per_folder = max_img_per_folder

        self.x_train, self.y_train = self.__load_data("train")
        self.x_test, self.y_test = self.__load_data("test")

    def init2(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    # def __add__(self, other):
    #     return DataLoader(
    #         x_train=(self.x_train + other.x_train),
    #         x_test=(self.x_test + other.x_test),
    #         y_train=(self.y_train + other.y_train),
    #         y_test=(self.y_test + other.y_test),
    #     )

    def __load_data(self, mode: str) -> (list, list):
        """
        :param mode: must be equal to "train" if you want train data and "test" if you want test data
        :return: None
        """

        # verify which  mode it is, give right parameters and raise an exception if the mode isn't good
        directory, path, labels_limit, labels, x, y = (
            (
                self.__train_dir,
                self.__train_path,
                self.__labels_limit_train,
                self.__labels_train,
                [],
                [],
            )
            if mode == "train"
            else (
                self.__test_dir,
                self.__test_path,
                self.__labels_limit_test,
                self.__labels_test,
                [],
                [],
            )
            if mode == "test"
            else (None, None, None, None, None, None)
        )

        if None in (directory, path, labels_limit, labels, x, y):
            raise Exception("you must specify test or train to get the correct data")

        # we recuperate all the filename of all images and then
        # we crop if there is a limit of integration of data
        for dataset in range(len(directory)):
            img_list = os.listdir(path + "/" + directory[dataset])
            if self.__max_img_per_folder is not None:
                img_list = img_list[0 : self.__max_img_per_folder]

            print("Loaded the images of dataset-" + "{}".format(directory[dataset]))
            # we read all images, and we append it into x
            nbr_img = 0
            for img in img_list:
                input_img = cv2.imread(path + "/" + directory[dataset] + "/" + img)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img = np.array(
                    cv2.resize(input_img, (self.__shape[0], self.__shape[1]))
                )
                input_img = np.array(input_img)
                # input_img = input_img.reshape(224, 224, 1)
                input_img = np.array(expand_dims(input_img, axis=2))
                x.append(input_img)
                nbr_img += 1
                labels_limit.append(nbr_img)
                y.append(self.get_id_by_name(directory[dataset]))

        x = np.array(x).astype(np.float32) / 255

        nbr_samples = x.shape[0]
        labels = np.ones((nbr_samples,), dtype=np.int64)
        acc, value = 0, 0
        for id in labels_limit:
            labels[acc:id] = value
            acc = id + 1
            value += 1

        return np.array(x), np.array(y)

    def get_label_name_by_id(self, id: int) -> str:
        return self.__labels_name[id]

    def get_id_by_name(self, name: str) -> int:
        return self.__labels_name.index(name)

    def get_train_data(self):
        print(
            "x_train:",
            len(self.x_train),
            "\ny_train:",
            len(self.y_train),
            "\nx_test:",
            len(self.x_test),
            "\ny_test:",
            len(self.y_test),
        )

        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test
