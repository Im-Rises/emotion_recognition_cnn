import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img

df = pd.read_csv("affectnet.csv")

X = df.image
y = df.emotion


for image in X:
    plt.imshow(img.imread(image))
    plt.show()