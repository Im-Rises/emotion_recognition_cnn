import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image
from keras_preprocessing.image import load_img
from sklearn.model_selection import train_test_split

from CnnModel import CnnModel
import handle_dataset as hdtst
from random import randrange

if __name__ == "__main__":
    # CKPLUS variables
    saved_model_name = "cnn_ckplus_model"
    test_size = 0.2
    images_shape = (48, 48, 1)
    images_path = "../../Databases/CKPLUS/CK+48/"

    # Create dictionaries
    value_emotion_dic = hdtst.create_dictionary(images_path)

    # ---------------------------------------------------------------------------------------------------
    # # Create csv dataset if they do not exist
    # if not (exists(train_csv_file) and exists(test_csv_file)):
    #     hdtst.create_csv_from_dataset(train_csv_file, "emotion;pixels", train_path)
    # print("Dataset create or already created")
    #
    # # Load csv dataset
    # X_train, y_train = hdtst.load_csv_dataset(train_csv_file, "pixels", "emotion")
    # print("Load dataset from csv")
    # ---------------------------------------------------------------------------------------------------

    # Load data
    X, y = hdtst.load_dataset(images_path)
    # Preprocess images and image label
    X, y = hdtst.preprocess_images(X, y, images_shape)
    # Split datas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Create CNN
    cnn = CnnModel(images_shape, saved_model_name)
    # Fit and predict
    cnn.fit(X_train, y_train, X_test, y_test)
    # Predict one image
    random_image_index = randrange(len(X_test))
    cnn.predict_image(
        X_test[random_image_index], y_test[random_image_index], value_emotion_dic
    )
    # Save model and weights
    cnn.save(saved_model_name)

    my_X_test, my_y_test = hdtst.load_dataset_v2("../../Databases/My_test/")
    random_image_index = randrange(len(my_X_test))
    my_X_test, my_y_test = hdtst.preprocess_images(my_X_test, my_y_test, images_shape)
    cnn.predict_image(
        my_X_test[random_image_index], my_y_test[random_image_index], value_emotion_dic
    )
