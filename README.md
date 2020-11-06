# Tensorflow Code for the paper: 
*[S-VVAD: Visual Voice Activity Detection by Motion Segmentation](Link)

## Overview
![BlockDiagram](https://github.com/muhammadshahidwandar/S-VVAD/blob/master/images/Fig_Main.jpg)

The Method consist of following steps as shown in figure above

1. Training a ResNet50 model with the pre-trained weights used for network initialization. Any framework such as tensorflow, pytorch or Caffe can be used. We used code from (RealVAD)(https://github.com/muhammadshahidwandar/Visual-VAD-Unsupervised-Domain-Adaptation) for this step.  

2. Class activation map generation using Gradient-CAM for Voice Activity Detection (VAD) labels: 0: not-speaking, 1: speaking. We used code from (https://github.com/insikk/Grad-CAM-tensorflow).

3. VAD-motion-cues-based mask generation.
 
4. Fully Convolution Network (FCN) training using VAD-cues' generated Masks in step 3.

5. Testing Fully Convolution Network (FCN)  using test dynamic images and saving those masks.

6. Bounding Box Generation around speaking and notspeaking segmented Cues. We used from (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html) for this step.

## Sub-directories and Files
There are four sub-directories described as follows:

### images
Containes over all training block diagram and some sample images of intermediate stages such as CAMs for speaking and not-speaking overlayed on Dynamic image and Raw CAMs as well, mask generation images.

### VAD-Mask-Generation
Containes some sample train and validation set for RealVAD dataset as explained in S-VVAD paper.  

### FCN-Training

``FCN_Train_Main``: To train Fully Convolutional ResNet-50 model on a given dataset 

``resnet_fcn.py``: Resnet-based FCN model defienation 

``datageneratorRealVAD.py``: Image batch generator with segmentation mask and BB from each Image

``datageneratorTest.py``: Sequential image batch generator with only BB annoation

### FCN-Testing

``TestFCN_Main``: To Reload and test the trained FCN model on a test set and saving the generated masks
``datageneratorTest.py``: Test image batch geneartor 

Some pre-trained ResNet50 model for tensorflow can be downloaded from this link (https://drive.google.com/drive/folders/1dHYOMuzXHL46P1zDgDyDj9NgYzV1nNSS?usp=sharing)

## Dependencies
* Python 3.5
* Tensorflow 1.12
* Opencv 3.0
* Natsort 7.0.1
* scipy  0.16.0


## How it works
1- Obtain your target datasets e.g.  RealVAD Dataset (https://github.com/IIT-PAVIS/Voice-Activity-Detection)

2- Generate and save the dynamic image by using (https://github.com/hbilen/dynamic-image-nets) 

3- Define your training and test folds in the text files (example files given as trainRealVAD1.txt and testRealVAD1.txt in ValidationFold sub-directory)

4- Change paths and parameters in FCN_Train_Main.py to train ResNet model

5- Test model on test set by using Model_Evaluation.py and BB projection on speaking Notspeaking segmentated.


## Reference

**S-VVAD: Visual Voice Activity Detection by Motion Segmentation**  
Muhammad Shahid,Cigdem Beyan and Vittorio Murino, IEEE Winter Conference on Applications of Computer Vision (WACV) 2021
```
@inproceedings{shahid2019SVVAD,
  title={Visual Voice Activity Detection by Motion Segmentation},
  author={Shahid, Muhammad and Beyan, Cigdem and Murino, Vittorio},
  booktitle={2021 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={0--0},
  year={2021}
}
