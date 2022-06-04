import cv2
import numpy as np
import pandas as pd
import requests


def read_img_from_url(url):
    resp = requests.get(url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # for testing
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_csv(str):
    str = str.split(",")
    image1 = str[:4]
    image2 = str[5:9]
    image3 = str[10:14]
    analyses = str[15:]
    print(image1)
    print(image2)
    print(image3)
    print(analyses)
    print("\n")
    url = "http://farm4.staticflickr.com/3679/12137399835_d9075d3194_b.jpg"
    # read_img_from_url()


def read_csv(filepath):
    image_already_processed = []
    with open(filepath, "r") as file:
        parse_csv(file.readline())
        parse_csv(file.readline())


read_csv("faceexp-comparison-data-test-public.csv")
