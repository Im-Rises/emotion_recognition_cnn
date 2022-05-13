import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

import handle_dataset as hdtst
from keras.applications.vgg16 import VGG16

images_shape = (224, 224, 3)

model = VGG16(
    weights=None, input_shape=images_shape
)  # Création du modèle VGG-16 implementé par Keras
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

train_path = "../../Databases/FER-2013/train/"
# train_path = "../../Databases/My_Test/"
test_path = "../../Databases/FER-2013/test/"
# test_path = "../../Databases/My_Test/"

X_train, y_train = hdtst.load_dataset_v2(train_path, images_shape)
X_train = hdtst.preprocess_images(X_train, images_shape)
print("Train data loaded")
X_test, y_test = hdtst.load_dataset_v2(test_path, images_shape)
X_test = hdtst.preprocess_images(X_test, images_shape)
print("Tests data loaded")

# early_stop = EarlyStopping(monitor="val_loss", patience=2)
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=25,
)
