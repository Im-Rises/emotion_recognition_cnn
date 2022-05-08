from CnnModel import CnnModel
import handle_dataset as hdtst

if __name__ == "__main__":
    train_path = "../../Databases/FER-2013/train/"
    test_path = "../../Databases/FER-2013/test/"
    saved_model_name = "cnn_fer_model"
    number_of_emotion = 7
    images_shape = (48, 48, 1)

    # Create dictionaries
    value_emotion_dic, emotion_value_dic = hdtst.create_dictionary(train_path)
    # Load data
    X_train, y_train = hdtst.load_dataset(train_path)
    X_test, y_test = hdtst.load_dataset(test_path)
    # Preprocess images and image label
    X_train, y_train = hdtst.preprocess_images(X_train, y_train, images_shape)
    X_test, y_test = hdtst.preprocess_images(X_test, y_test, images_shape)
    print(y_train[24000])

    cnn = CnnModel(number_of_emotion, images_shape, "")
    cnn.fit(X_train, y_train, X_test, y_test)
    cnn.predict_image(X_test[0], y_test[0], value_emotion_dic)
    cnn.save(saved_model_name)
