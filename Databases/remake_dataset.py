import os
import shutil

import cv2
import numpy as np
import pandas as pd


def get_best_emotion(list_of_emotions, emotions):
    best_emotion = np.argmax(emotions)
    if best_emotion == "neutral" and sum(emotions[1::]) > 0:
        emotions[best_emotion] = 0
        best_emotion = np.argmax(emotions)
    return list_of_emotions[best_emotion]


def read_and_clean_csv(path):
    # we read the csv and we delete all the rows which contains NaN
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def rewrite_image_from_df(df):
    print("Moving images from FERPlus inside FER-2013")
    # we setup an accumulator to print if we have finished a task
    acc = ""
    emotions = [
        "neutral",
        "happy",
        "surprise",
        "sad",
        "angry",
        "disgust",
        "fear",
        "contempt",
        "unknown",
        "NF",
    ]

    # we rewrite all the image files
    for row in range(len(df)):
        item = df.iloc[row]
        if item["Usage"] not in ["", acc]:
            print(f"{item['Usage']} done")
        if (item['Usage'] == "Training"):
            image = cv2.imread(f"./FERPlus/output/FER2013Train/{item['Image name']}")
        elif item['Usage'] == "PublicTest":
            image = cv2.imread(f"./FERPlus/output/FER2013Valid/{item['Image name']}")
        else:
            image = cv2.imread(f"./FERPlus/output/FER2013Test/{item['Image name']}")
        acc = item["Usage"]
        if acc == "Training":
            cv2.imwrite(
                f"./FER-2013/train/{get_best_emotion(emotions, item[2::])}/{item['Image name']}",
                image,
            )
        else:
            cv2.imwrite(
                f"./FER-2013/test/{get_best_emotion(emotions, item[2::])}/{item['Image name']}",
                image,
            )


if __name__ == "__main__":
    os.system('python ./FERPLUS/src/generate_training_data.py -d ./FERPLUS/output -fer ./FER-2013/fer2013.csv -ferplus ./FERPLUS/fer2013new.csv')
    df = read_and_clean_csv("./FERPlus/fer2013new.csv")
    rewrite_image_from_df(df)
