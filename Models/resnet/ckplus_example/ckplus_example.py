import os

import cv2
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.utils import np_utils
from keras.utils.version_utils import callbacks
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data_path = "../../../Databases/CKPLUS/ck/CK+48/"
data_dir_list = os.listdir(data_path)

img_rows = 256
img_cols = 256
num_channel = 1

num_epoch = 10

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + "/" + dataset)
    print("Loaded the images of dataset-" + "{}\n".format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + "/" + dataset + "/" + img)
        # input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (48, 48))
        img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype("float32")
img_data = img_data / 255
num_classes = 7

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype="int64")

labels[0:134] = 0  # 135
labels[135:188] = 1  # 54
labels[189:365] = 2  # 177
labels[366:440] = 3  # 75
labels[441:647] = 4  # 207
labels[648:731] = 5  # 84
labels[732:980] = 6  # 249

names = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]


def get_label(id):
    return ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"][id]


Y = np_utils.to_categorical(labels, num_classes)

# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=2
)
x_test = X_test

input_shape = (48, 48, 3)

model = Sequential()
model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])

filename = "model_train_new.csv"
filepath = "Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

csv_log = callbacks.CSVLogger(filename, separator=",", append=False)
checkpoint = callbacks.ModelCheckpoint(
    filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)
callbacks_list = [csv_log, checkpoint]
callbacks_list = [csv_log]

hist = model.fit(
    X_train,
    y_train,
    batch_size=7,
    epochs=50,
    verbose=1,
    validation_data=(X_test, y_test),
    callbacks=callbacks_list,
)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test accuracy:", score[1])

test_image = X_test[0:1]
print(test_image.shape)


print(model.predict(test_image))
# print(model.predict_classes(test_image)) # deprecated
print(y_test[0:1])

res = np.argmax(model.predict(X_test[9:18]), axis=-1)
plt.figure(figsize=(10, 10))

for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_test[i], cmap=plt.get_cmap("gray"))
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel("prediction = %s" % get_label(res[i]), fontsize=14)
# show the plot
plt.show()

train_loss = hist.history["loss"]
val_loss = hist.history["val_loss"]
train_acc = hist.history["accuracy"]
val_acc = hist.history["val_accuracy"]

epochs = range(len(train_acc))

plt.plot(epochs, train_loss, "r", label="train_loss")
plt.plot(epochs, val_loss, "b", label="val_loss")
plt.title("train_loss vs val_loss")
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs, train_acc, "r", label="train_acc")
plt.plot(epochs, val_acc, "b", label="val_acc")
plt.title("train_acc vs val_acc")
plt.legend()
plt.figure()
plt.show()
