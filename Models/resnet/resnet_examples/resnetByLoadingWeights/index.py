import os

from keras import Model
from keras.applications.resnet import ResNet50
from keras.callbacks import ModelCheckpoint, TensorBoard

from emotion_recognition_cnn.resnet.DataLoader import DataLoader

train_path = "./../../../../Databases/FER-2013/train/"
test_path = "./../../../../Databases/FER-2013/test/"

fer2013 = DataLoader(
    train_path=train_path,
    test_path=test_path,
    labels=os.listdir(train_path),
    max_img_per_folder=50,
    shape=(224, 224),
)

x_train, y_train = fer2013.get_train_data()
x_test, y_test = fer2013.get_test_data()

name = "resnet50"

checkpoint_path = "checkpoints/" + name + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
os.system("mkdir {}".format(checkpoint_dir))

# save model after each epoch
cp_callback = ModelCheckpoint(filepath=checkpoint_path, verbose=1)
tensorboard_callback = TensorBoard(log_dir="tensorboard_logs/" + name, histogram_freq=1)

resnet50 = ResNet50(
    # input_shape=(224, 224, 1),
    # classes=7,
    # include_top=False,
    weights="imagenet",
)

model = Model(inputs=resnet50.input, outputs=resnet50.get_layer(index=-2).output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    x=x_train,
    y=y_train,
    epochs=20,
    shuffle=True,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[cp_callback, tensorboard_callback],
)
