import os
import shutil

import cv2
import numpy as np
import pandas as pd


def get_best_emotion(list_of_emotions, emotions):
    return list_of_emotions[np.argmax(emotions)]


def read_and_clean_csv(path):
    # we read the csv and we delete all the rows which contains NaN
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def rewrite_image_from_df(df):
    # we setup an accumulator to print if we have finished a task
    acc = ""

    # we rewrite all the image files
    for row in range(len(df)):
        item = df.iloc[row]
        if item["Usage"] not in ["", acc]:
            print(f"{item['Usage']} done")
        image = cv2.imread(f"{item['Usage']}/{item['Image name']}")
        acc = item["Usage"]
        if acc == "Training":
            cv2.imwrite(
                f"../FER-2013/train/{get_best_emotion(df.keys()[2::], item[2::])}/{item['Image name']}",
                image,
            )
        else:
            cv2.imwrite(
                f"../FER-2013/test/{get_best_emotion(df.keys()[2::], item[2::])}/{item['Image name']}",
                image,
            )


if __name__ == "__main__":
    df = read_and_clean_csv("fer2013new.csv")
    rewrite_image_from_df(df)
