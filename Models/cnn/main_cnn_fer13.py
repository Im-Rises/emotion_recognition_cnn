from Models.cnn.cnn_model import CnnModel
from Models import handle_dataset as hdtst
from random import randrange

if __name__ == "__main__":
    # Fer-13 variables
    train_path = "../../Databases/FER-2013/train/"
    test_path = "../../Databases/FER-2013/test/"
    train_csv_file = "train_fer.csv"
    test_csv_file = "test_fer.csv"
    saved_model_name = "cnn_fer_model"
    number_of_emotions = 7
    images_shape = (48, 48, 1)

    # Create dictionaries
    value_emotion_dic = hdtst.create_dictionary(train_path)

    # ---------------------------------------------------------------------------------------------------
    # # Create csv dataset if they do not exist
    # if not (exists(train_csv_file) and exists(test_csv_file)):
    #     hdtst.create_csv_from_dataset(train_csv_file, "emotion;pixels", train_path)
    #     hdtst.create_csv_from_dataset(test_csv_file, "emotion;pixels", test_path)
    # print("Dataset create or already created")
    #
    # # Load csv dataset
    # X_train, y_train = hdtst.load_csv_dataset(train_csv_file, "pixels", "emotion")
    # X_test, y_test = hdtst.load_csv_dataset(test_csv_file, "pixels", "emotion")
    # print("Load dataset from csv")
    # ---------------------------------------------------------------------------------------------------

    # Load dataset
    X_train, y_train = hdtst.load_dataset(train_path, images_shape)
    X_test, y_test = hdtst.load_dataset(test_path, images_shape)

    # Preprocess images and image label
    X_train = hdtst.preprocess_images(X_train, images_shape)
    X_test = hdtst.preprocess_images(X_test, images_shape)

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
