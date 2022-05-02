# emotion_recognition_cnn

## Description

Deep Learning Project made in python for the recognition of emotion of a person.

Our goal is to obtein a minimum 90% accuracy on emotion recognition from any pictures.
The app need to be able to analyse facial expression dicretly by analysing one image or by analysing frame by frame a video.  
Finding all visbiles faces and give the current emotion of the person base on the six main facial expressions :

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

- Analyse images/videos and foud every faces.
- Display emotion of a person based on facial expression.

## Report

The report is written in french. No other languages are available.  
[Word document link](https://esmefr-my.sharepoint.com/:w:/g/personal/clement_reiffers_esme_fr/EQLW0WK_l6hHrJRBIOaRYeQBrQLS2fZTjtCm68l-NXpW_g?e=4%3ARP8DM1&at=9&CID=D924432C-3B7E-4D12-B1AF-5F9A98207FC7&wdLOR=c46E7383C-126E-40A3-BA99-964061BF8370)

## Documentations

<https://www.youtube.com/watch?v=u7obuspdQu4>  
<https://www.datacorner.fr/reco-faciale-opencv/>  

## Installation

### CUDA and co installation

Before using the program, you should install CUDA and SDK to allow the program to run with the GPU and not the CPU.
The app is asking a lot of processing, so to speed it we use the GPU instead of the CPU.  

Check compatible version with your tensorflow :  
<https://www.tensorflow.org/install/source#gpu>  
<https://www.tensorflow.org/install/source_windows>  

Visual Studio :  
<https://visualstudio.microsoft.com/fr/downloads/>

CUDA :  
<https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/>  

cuDNN :  
<https://developer.nvidia.com/cudnn>

Tensorflow :  
<https://www.tensorflow.org/install/gpu>

### Version of libs used

- Visual Studio 2022
- CUDA 11.2
- cuDNN 8.1
- tensorflow_gpu-2.6.0
- keras 2.6

!!! Tensorflow install keras 2.7 or higher, make sure to reinstall keras 2.6 !!!

## Databases

[Google facial expression comparison dataset](https://research.google/tools/datasets/google-facial-expression/)  
[AffectNet-Sample](https://www.kaggle.com/mouadriali/affectnetsample)  
[CKPLUS](https://www.kaggle.com/shawon10/ckplus)  
[FER-2013](https://www.kaggle.com/msambare/fer2013)  

## Architectures

[Resnet](https://www.kaggle.com/datasets/keras/resnet50/code?resource=download)  

## Collaborateurs

Quentin MOREL :

- @Im-Rises
- <https://github.com/Im-Rises>

Clément REIFFERS :  

- @clementreiffers  
- <https://github.com/clementreiffers>

Yohan COHEN-SOLAL :

- @YohanCohen-Solal  
- <https://github.com/YohanCohen-Solal>
