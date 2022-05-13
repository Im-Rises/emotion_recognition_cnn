from sklearn.model_selection import train_test_split

from CnnModel import CnnModel
from emotion_recognition_cnn import handle_dataset as hdtst
from random import randrange

if __name__ == "__main__":
    # Fer-13 variables
    train_path = "../../Databases/FER-2013/train/"
    test_path = "../../Databases/FER-2013/test/"
    saved_model_name = "cnn_fer_model"
    number_of_emotion = 7
    images_shape = (48, 48, 1)

    # Create dictionaries
    value_emotion_dic = hdtst.create_dictionary(train_path)
    # Load data
    X_train, y_train = hdtst.load_dataset(train_path)
    X_test, y_test = hdtst.load_dataset(test_path)
    # Preprocess images and image label
    X_train, y_train = hdtst.preprocess_images(X_train, y_train, images_shape)
    X_test, y_test = hdtst.preprocess_images(X_test, y_test, images_shape)

    # Create CNN
    cnn = CnnModel(number_of_emotion, images_shape, saved_model_name)
    # Fit and predict
    cnn.fit(X_train, y_train, X_test, y_test)
    # Predict one image
    random_image_index = randrange(len(X_test))
    cnn.predict_image(
        X_test[random_image_index], y_test[random_image_index], value_emotion_dic
    )
    # Save model and weights
    cnn.save(saved_model_name)

    # ----------------------------------------------------------------------------------------------------------------

    # CKPLUS variables
    saved_model_name = "cnn_ckplus_model"
    number_of_emotion = 7
    test_size = 0.2
    images_shape = (48, 48, 1)
    images_path = "../../Databases/CKPLUS/CK+48/"

    # Create dictionaries
    value_emotion_dic = hdtst.create_dictionary(images_path)
    # Load data
    X, y = hdtst.load_dataset(images_path)
    # Preprocess images and image label
    X, y = hdtst.preprocess_images(X, y, images_shape)
    # Split datas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Create CNN
    cnn = CnnModel(number_of_emotion, images_shape, saved_model_name)
    # Fit and predict
    cnn.fit(X_train, y_train, X_test, y_test)
    # Predict one image
    random_image_index = randrange(len(X_test))
    cnn.predict_image(
        X_test[random_image_index], y_test[random_image_index], value_emotion_dic
    )
    # Save model and weights
    cnn.save(saved_model_name)
