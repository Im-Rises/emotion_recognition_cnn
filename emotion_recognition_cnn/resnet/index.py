from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

from DataLoader import DataLoader
from resnet_test import *

input_shape = (None, 32, 32, 3)

affecnet = DataLoader(
    path="../../Databases/AffectNet/train_class",
    labels=[
        "neutral",
        "happy",
        "angry",
        "sad",
        "fear",
        "surprise",
        "disgust",
        "contempt",
    ],
    labels_limit=[2500 * x for x in range(1, 9)],
    max_label=2500,
)

x_train, x_test, y_train, y_test = affecnet.train_test_data()

# ckplus48 = DataLoader(
#     path="../../Databases/CKPLUS/ck/CK+48",
#     labels=["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"],
#     labels_limit=[134, 188, 365, 440, 647, 731, 980],
#     shape=input_shape,
# )
#
# x_train, x_test, y_train, y_test = ckplus48.train_test_data()

try:
    resnet18 = load_model("savedModel")
    resnet18.summary()
except:
    """
    # DATA AUGMENTATION
    """
    # aug = ImageDataGenerator(
    #     horizontal_flip=True, width_shift_range=0.05, height_shift_range=0.05
    # )
    # aug.fit(x_train)

    """
    # INIT RESNET
    """
    resnet18 = create_resnet(8, input_shape)

    """
    # ENTRAINEMENT
    """
    es = EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy")
    # I did not use cross validation, so the validate performance is not accurate.
    STEPS = len(x_train) / 256
    history = resnet18.fit(
        # aug.flow(x_train, y_train, batch_size=256),
        steps_per_epoch=STEPS,
        batch_size=256,
        x=x_train,
        y=y_train,
        epochs=200,
        validation_data=(x_test, y_test),
        callbacks=[es],
    )

    resnet18.save("savedModel")

pred = np.argmax(resnet18.predict(x_test[0]), axis=-1)
plt.imshow(x_test[0].reshape(32, 32, 3))
plt.title(affecnet.get_label(pred[0]))
plt.show()

for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_test[i][0], cmap=plt.get_cmap("gray"))
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel(
        "%s" % affecnet.get_label(np.argmax(resnet18.predict(x_test[i])[0])),
        fontsize=14,
    )
# show the plot
plt.show()
