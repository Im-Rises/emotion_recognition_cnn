# emotion_recognition_cnn

<p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="pythonLogo" style="height:50px">
    <img src="https://user-images.githubusercontent.com/59691442/169644815-7c59a948-09a4-4cd5-9a7d-d06d5dcd3ce1.svg" alt="tensorflowLogo" style="height:50px;">
    <img src="https://user-images.githubusercontent.com/59691442/169644811-109bf300-c795-44c4-8bab-f6900f97c422.png" alt="kerasLogo" style="height:50px;">
</p>

## Description

Deep Learning AI Emotion recognition made in python with Tensorflow/Keras.

### Expression recognition

<img src="readme_files/demo1.gif"/>

The app need to be able to analyse facial expression dicretly by analysing one image or by analysing frame by frame a video.  
Finding all visibles faces and show the current emotion state of the person base on the 7 main facial expressions :

- Neutral/Normal
- Sadness
- Happiness
- Fear
- Anger
- Surprise
- Disgust

The app is using the CNN (Convolutional neural network) with the ResNet50 Architecture via Transfer Learning.
The AI is trained with the FER-2013 and FERPLUS datasets allowing it to understand how to analyse a person's emotion from a picture.

### Features

The app features :

- UI
- Handle face detection
- Analyse emotions of a person

### Report and Jupyter Notebook

A report is available, but it is only in french.

[Word document link](https://esmefr-my.sharepoint.com/:w:/g/personal/clement_reiffers_esme_fr/EQLW0WK_l6hHrJRBIOaRYeQBrQLS2fZTjtCm68l-NXpW_g?e=4%3ARP8DM1&at=9&CID=D924432C-3B7E-4D12-B1AF-5F9A98207FC7&wdLOR=c46E7383C-126E-40A3-BA99-964061BF8370)

Thre is also a Jupyter Notebook named `demo.ipynb` at the root folder of the project.

## Screenshots and videos

### Images

Placeholder

### Videos

Placeholder

---

## Installation

### Quickstart

Firstly you need to install python. I recommand you the python 3.6-3.8.

If you just want to start the UI app, then just follow the `1. Python, Tensorflow, Keras` instructions just below.

In the case you want to test the models and train them. I would advised you to follow the `1. Python, Tensorflow, Keras` instructions below and the second set of instructions `2. CUDA and cuDNN installation`. **You will need a good GPU to train the models** if you don't want the training to take more than 2 hours.

Once everything is installed, go to part `3. Train a model and use it` to train a model and test it.

### 1. Python, Tensorflow, Keras, OpenCV

To use the program, you need to install python first (version 3.8 advised).
<https://www.python.org>

You can than install the required python packages. They are all listed in the requiremetns.txt file.
To install them all directy, type the following command:

```terminal
pip install -r requirements.txt
```

With all the packages installed you'll be able to start the `app.py` file at the root of the project, it will start the HIM. Once the HIM is started go in your browser to this address `http://127.0.0.1:3134` allow the localhost website to use your camera and have fun.

If you don't want to use your browser, you can use the python UI version in `emotion_recognition/prediction.py`.

### 2. CUDA and cuDNN installation

Before using the program, you should install CUDA and SDK to allow the program to run with the GPU and not the CPU.
The app is asking a lot of processing, so to speed it we use the GPU instead of the CPU.  

While programming we used different versions of tensorflow, CUDA etc... To know which version of Tensorflow use with your version of CUDA, cudNN etc... check the following website.  

<https://www.tensorflow.org/install/source#gpu>

#### Windows

Follow this tutorial from Tensorflow :  
<https://www.tensorflow.org/install/source_windows#install_gpu_support_optional>

Visual Studio or redistribuable :  
<https://visualstudio.microsoft.com/fr/downloads/>  
<https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170>

CUDA :  
<https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/>  

cuDNN :  
<https://developer.nvidia.com/cudnn>

Tensorflow :  
<https://www.tensorflow.org/install/gpu>

Follow this video steps if you have difficulties <https://www.youtube.com/watch?v=hHWkvEcDBO0&list=LL>.  
If you are unable to install CUDA and CuDNN, I would advise you to use a TPU, by using Google Collab.
<https://colab.research.google.com>

Once you have installed the ncessary packages, app, SDK, etc... You need to download the FER-13 dataset in the `3. Download the FER-13 database` section.  
You can then use start the python console script in emotion_recognition/Models/training.py.

#### Linux

Follow this tutorial from Tensorflow :  
<https://www.tensorflow.org/install/source#install_gpu_support_optional_linux_only>

Once you have installed the ncessary packages, app, SDK, etc... You need to download the FER-13 dataset in the `3. Download the FER-13 database` section.  
You can then use start the python console script in emotion_recognition/Models/training.py.

### 3. Train a model and use it

Once everything is installed you can start the training by starting the `training.py` script in emotion_recognition/Models/training.py.
There you can select which model you want to train by transfer learning between :

1. resnet50
2. vgg16
3. xception
4. inception_resnet_v2
5. inception_v3
6. resnet50v2
7. resnet101

If you want to use another model for the UIs. Save your model when asked by the promped and change the load_weights/load_model model in the app.py or prediction.py.

You can also change the database on which you're training. By default the AI is set to be trained on FER-2013 dataset that you need to download first.

If you want to use FERPlus for better performences, you will need to download FERPLUS and FER-2013. Extract them in the databases folder next to the `datasets.txt` file as two folder FER-2013 (containting train, test folders and fer2013.csv) and FERPlus folder containing all the FERPlus's microsoft repository.
Last step is to start the `remake_dataset.py` that will concatenate all FERPlus images inside the FER-2013 folder.

All datasets can be downloaded below!!!

---

## Databases

[FER-2013](https://www.kaggle.com/msambare/fer2013)  
[FERPLUS](https://github.com/microsoft/FERPlus)

## Libraries

Python :  
<https://www.python.org>

Tensorflow/Keras :  
<https://www.tensorflow.org>

OpenCV :  
<https://www.datacorner.fr/reco-faciale-opencv/>
<https://www.datacorner.fr/reco-faciale-opencv-2/>

Flask :  
<https://flask.palletsprojects.com/en/2.1.x/>

## Documentations

CNN, ANN, RNN presentation :  
<https://www.youtube.com/watch?v=u7obuspdQu4>

How to elaborate a CNN :  
<https://www.analyticsvidhya.com/blog/2021/11/facial-emotion-detection-using-cnn/>  
<https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/>  

Transfert learning :  
<https://www.datacorner.fr/vgg-transfer-learning/>

FERPLUS :  
@inproceedings{BarsoumICMI2016,
    title={Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},
    author={Barsoum, Emad and Zhang, Cha and Canton Ferrer, Cristian and Zhang, Zhengyou},
    booktitle={ACM International Conference on Multimodal Interaction (ICMI)},
    year={2016}
}

## Contributors

Quentin MOREL :

- @Im-Rises
- <https://github.com/Im-Rises>

Clément REIFFERS :  

- @clementreiffers  
- <https://github.com/clementreiffers>

Yohan COHEN-SOLAL :

- @YohanCohen-Solal  
- <https://github.com/YohanCohen-Solal>

[![GitHub contributors](https://contrib.rocks/image?repo=Im-Rises/emotion_recognition_cnn)](https://github.com/Im-Rises/emotion_recognition_cnn/graphs/contributors)
