import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img

emotions = {
    "class001": "neutral",
    "class002": "happy",
    "class003": "sad",
    "class004": "surprise",
    "class005": "fear",
    "class006": "disgust",
    "class007": "anger",
    "class008": "contempt"
}

plt.close()
df = pd.read_csv("affectnet.csv")
df = df[df["emotion"] == "class002"]


X = list(map(lambda im: img.imread(im), df.image[200:300]))
y = df.emotion[200:300]

print(X)


for i in range(len(X)):
    print(i)
    plt.imshow(X[i])
    plt.title("image numero {0}, class : {1}".format(i, emotions[y[i]]))
    plt.show()
