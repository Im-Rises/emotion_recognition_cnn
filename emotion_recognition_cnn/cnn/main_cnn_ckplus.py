from sklearn.model_selection import train_test_split

from CnnModel import CnnModel
from emotion_recognition_cnn import handle_dataset as hdtst
from random import randrange

if __name__ == "__main__":
    # CKPLUS variables
    saved_model_name = "cnn_ckplus_model"
    test_size = 0.2
    images_shape = (48, 48, 1)
    images_path = "../../Databases/CKPLUS/CK+48/"
    number_of_emotions = 7
    # images_path = "../../Databases/My_test/"

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
    X, y = hdtst.load_dataset(images_path, images_shape)
    # Preprocess images and image label
    X = hdtst.preprocess_images(X, images_shape)
    # Split datas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Create CNN
    cnn = CnnModel(number_of_emotions, images_shape, saved_model_name)
    # Fit and predict
    cnn.fit(X_train, y_train, X_test, y_test)
    # Predict one image
    random_image_index = randrange(len(X_test))
    cnn.predict_image(
        X_test[random_image_index], y_test[random_image_index], value_emotion_dic
    )
    # Save model and weights
    cnn.save(saved_model_name)

    my_X_test, my_y_test = hdtst.load_dataset("../../Databases/My_test/", images_shape)
    my_X_test = hdtst.preprocess_images(my_X_test, images_shape)
    for image, emotion in zip(my_X_test, my_y_test):
        cnn.predict_image(
            image,
            emotion,
            value_emotion_dic,
        )
