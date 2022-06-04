import pandas as pd


def parse_csv(str):
    print(str)


def read_csv(filepath):
    with open(filepath, "r") as file:
        parse_csv(file.readline())


df = pd.read_csv("faceexp-comparison-data-test-public.csv", lineterminator="\r")
