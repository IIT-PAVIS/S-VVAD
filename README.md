# Tensorflow Code for the paper "S-VVAD: Visual Voice Activity Detection by Motion Segmentation"

[Full Paper and Supplementary Material](https://openaccess.thecvf.com/content/WACV2021/html/Shahid_S-VVAD_Visual_Voice_Activity_Detection_by_Motion_Segmentation_WACV_2021_paper.html) <br />
[Extended Video Presentation](https://cbeyan.github.io/) <br />

## Overview
![BlockDiagram](https://github.com/muhammadshahidwandar/S-VVAD/blob/master/images/Fig_Main.jpg)

S-VVAD consists of the following steps as shown in the figure above:

1. Training a ResNet50 model with the pre-trained weights used for network initialization. 
Any framework such as tensorflow, pytorch or Caffe can be used. 
We used our previous code that can be found in: (https://github.com/muhammadshahidwandar/Visual-VAD-Unsupervised-Domain-Adaptation) for this step.  

2. Class activation map generation using Grad-CAM method (Selvaraju et al.) for Voice Activity Detection (VAD) labels: 
0: not-speaking, 1: speaking. 
Grad-CAM code can be found in: (https://github.com/insikk/Grad-CAM-tensorflow).

3. VAD-motion-cues-based mask generation.
 
4. Fully Convolution Network (FCN) training using the masks generated in Step 3.

5. Testing Fully Convolution Network (FCN) with multiple dynamic images and to saving the corresonding masks.

6. Bounding box generation which includes affinity propogation in test time.
The affinity propogation code can be found in: (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html).

## Sub-directories and Files
There are four sub-directories described as follows:

### images
Contains the block diagram of S-VVAD training, and some sample images of intermediate stages such as CAMs for speaking and not-speaking overlayed on dynamic images and raw CAMs as well, mask generation images.

### VAD-Mask-Generation
Contains some sample train and test sets for RealVAD dataset.  

### FCN-Training

``FCN_Train_Main``: To train Fully Convolutional ResNet-50 model on a given dataset 

``resnet_fcn.py``: ResBet-based Fully Convolutional ResNet-50 definition 

``datageneratorRealVAD.py``: Image batch generator including segmentation mask and bounding boxes

``datageneratorTest.py``: Sequential image batch generator with only bounding box annotation

### FCN-Testing

``TestFCN_Main``: To reload and test the trained FCN model on a test set and to save the generated masks

``datageneratorTest.py``: Image batch geneartor in test 

Pre-trained ResNet50 models for tensorflow can be downloaded from this link (https://drive.google.com/drive/folders/1dHYOMuzXHL46P1zDgDyDj9NgYzV1nNSS?usp=sharing)

## Dependencies
* Python 3.5
* Tensorflow 1.12
* Opencv 3.0
* Natsort 7.0.1
* scipy  0.16.0


## How it works
1- Obtain your target datasets, e.g.,  RealVAD Dataset (https://github.com/IIT-PAVIS/Voice-Activity-Detection)

2- Generate and save the multiple dynamic images by using (https://github.com/hbilen/dynamic-image-nets) 

3- Define your training and test folds as text files (example files are called: trainRealVAD1.txt and testRealVAD1.txt that can be found in the ValidationFold folder)

4- Change the paths and the parameters in FCN_Train_Main.py to train ResNet model

5- Test model by using Model_Evaluation.py and project the bounding boxes found as speaking/not-speaking by segmentation

## Modified Columbia dataset splits

As promised in the paper, we supply the group-based splits for Columbia datasets (so-called Modified Columbia dataset). 
The corresponding file is: ModifiedColumbia_splits.zip

## Reference

Muhammad Shahid, Cigdem Beyan and Vittorio Murino, "S-VVAD: Visual Voice Activity Detection by Motion Segmentation", 
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 2332-2341, 2021.
```
@InProceedings{Shahid_2021_WACV,
    author    = {Shahid, Muhammad and Beyan, Cigdem and Murino, Vittorio},
    title     = {S-VVAD: Visual Voice Activity Detection by Motion Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {2332-2341}
}
