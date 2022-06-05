import os

from keras.applications.resnet import preprocess_input
from keras.models import load_model
from matplotlib import pyplot as plt

from common_functions import (
    get_data,
    fit,
    evaluation_model,
    saveModel,
)

parameters = {
    "shape": [80, 80],
    "nbr_classes": 7,
    "train_path": "../../Databases/Affectnet/train_class",
    "test_path": "../../Databases/Affectnet/val_class",
    "batch_size": 8,
    "epochs": 50,
    "number_of_last_layers_trainable": 10,
    "learning_rate": 0.001,
    "nesterov": True,
    "momentum": 0.9,
}

train_files, test_files, train_generator, test_generator = get_data(
    preprocess_input=preprocess_input, parameters=parameters
)

model = load_model("trained_models/resnet50")

filename = "resnset50_fer2013&affecnet"

history = fit(
    model=model,
    train_generator=train_generator,
    test_generator=test_generator,
    train_files=train_files,
    test_files=test_files,
    parameters=parameters,
)
score = evaluation_model(model, test_generator)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 1.0])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()

if os.path.isfile(f"./logs/{filename}_parameters.log"):
    with open(f"./logs/{filename}_parameters.log", "r") as file:
        print(file.read())
        file.close()

choice = input("save model? (O/N)\n>>>")

if choice == "O":
    saveModel(filename=filename, model=model)
    with open(f"./logs/{filename}_parameters.log", "w") as file:
        file.write(f"{parameters}\nval_acc: {val_acc}\nval_loss: {val_loss}")
        file.close()
