from sklearn.model_selection import train_test_split

from zfnet import ZFNET
import handle_dataset as hdtst
from random import randrange
from os.path import exists

import time

if __name__ == "__main__":
    # Fer-13 variables
    train_path = "../../../Databases/AffectNet/train_class/"
    test_path = "../../Databases/AffectNet/test_class/"
    saved_model_name = "zfnet_fer_model"
    number_of_emotion = 8
    images_shape = (224, 224, 3)

    # Create dictionaries
    value_emotion_dic = hdtst.create_dictionary(train_path)

    X_train, y_train = hdtst.load_dataset(train_path)
    X_test, y_test = hdtst.load_dataset(test_path)
    print("Done")

    X_train = X_train.reshape(37553, 224, 224, 3)
    X_test = X_test.reshape(4000, 224, 224, 3)

    # Create CNN
    cnn = ZFNET(number_of_emotion, images_shape, saved_model_name)
    # Fit and predict
    cnn.fit(X_train, y_train, X_test, y_test)
    # Predict one image
    random_image_index = randrange(len(X_test))
    cnn.predict_image(
        X_test[random_image_index], y_test[random_image_index], value_emotion_dic
    )
    # Save model and weights
    cnn.save(saved_model_name)
