# emotion_recognition_cnn

<p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="pythonLogo" style="height:50px">
    <img src="https://user-images.githubusercontent.com/59691442/169644815-7c59a948-09a4-4cd5-9a7d-d06d5dcd3ce1.svg" alt="tensorflowLogo" style="height:50px;">
    <img src="https://user-images.githubusercontent.com/59691442/169644811-109bf300-c795-44c4-8bab-f6900f97c422.png" alt="kerasLogo" style="height:50px;">
</p>

## Description

### Expression recognition

Deep Learning Project made in python for the recognition of emotion of a person.

Our goal is to obtein a minimum 90% accuracy on emotion recognition from any pictures.
The app need to be able to analyse facial expression dicretly by analysing one image or by analysing frame by frame a video.  
Finding all visbiles faces and give the current emotion of the person base on the six main facial expressions :

- Neutral/Normal
- Sadness
- Happiness
- Fear
- Anger
- Surprise
- Disgust

The app is using the CNN (Convolutional neural network) to analyse the emotion of a person.
It will learn from different databases that contains several images of persons showing one of the six main emotion.
The AI learn from this database by analysing each image, once done our model is able to analyse faces images out of the database.

## Features

The app features :

- UI
- Handle images and camera video
- Multiple face detection and emotion analysis

### Report

The report is written in french. No other languages are available.  
[Word document link](https://esmefr-my.sharepoint.com/:w:/g/personal/clement_reiffers_esme_fr/EQLW0WK_l6hHrJRBIOaRYeQBrQLS2fZTjtCm68l-NXpW_g?e=4%3ARP8DM1&at=9&CID=D924432C-3B7E-4D12-B1AF-5F9A98207FC7&wdLOR=c46E7383C-126E-40A3-BA99-964061BF8370)

## Screenshots and videos

### Images

Placeholder

### Videos

Placeholder

## Installation

### Quickstart

If you just want to start the UI app, then just follow the `1. Python, Tensorflow, Keras` instructions just below.

In the case you want to test the models and train them. I would advised you to follow the `1. Python, Tensorflow, Keras` instructions below and the second set of instructions `2. CUDA and cuDNN installation`. **You will need a good GPU to train the models** if ypu don't want the training to take more than 2 hours.  

### 1. Python, Tensorflow, Keras, OpenCV

To use the program, you need to install python first (version 3.8 advised).
<https://www.python.org>

You can than install the required python packages. They are all listed in the requiremetns.txt file.
To install them all directy, type the following command:

```terminal
pip install -r requirements.txt
```

With all the packages installed you'll be able to start the main.py file at the root of the project, it will start the IHM from there you can input an image or use a camera to analysis your emotions.

### 2. CUDA and cuDNN installation

Before using the program, you should install CUDA and SDK to allow the program to run with the GPU and not the CPU.
The app is asking a lot of processing, so to speed it we use the GPU instead of the CPU.  

While programming we used different versions of tensorflow, CUDA etc... To know which version of Tensorflow use with your version of CUDA, cudNN etc... check the following website.  
<https://www.tensorflow.org/install/source#gpu>

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

### 3. Download the FER-13 database

The FER-13 dataset can be download below.
[FER-2013](https://www.kaggle.com/msambare/fer2013)

Once downloaded extract it and put here in the Database folder.

## Contributors

[![GitHub contributors](https://contrib.rocks/image?repo=Im-Rises/emotion_recognition_cnn)](https://github.com/Im-Rises/emotion_recognition_cnn/graphs/contributors)

Quentin MOREL :

- @Im-Rises
- <https://github.com/Im-Rises>

Clément REIFFERS :  

- @clementreiffers  
- <https://github.com/clementreiffers>

Yohan COHEN-SOLAL :

- @YohanCohen-Solal  
- <https://github.com/YohanCohen-Solal>

## Databases

[CKPLUS](https://www.kaggle.com/shawon10/ckplus)  
[FER-2013](https://www.kaggle.com/msambare/fer2013)  

## Documentations

CNN, ANN, RNN presentation :  
<https://www.youtube.com/watch?v=u7obuspdQu4>

How to elaborate a CNN :  
<https://www.analyticsvidhya.com/blog/2021/11/facial-emotion-detection-using-cnn/>  
<https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/>  

Transfert learning :  
<https://www.datacorner.fr/vgg-transfer-learning/>

OpenCV :  
<https://www.datacorner.fr/reco-faciale-opencv/>
<https://www.datacorner.fr/reco-faciale-opencv-2/>
