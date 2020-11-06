import tensorflow as tf
import numpy as np
import resnet_fcn
from datageneratorClusterLblMask import ImageDataGenerator2
import time
import  scipy
import  cv2
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from scipy.misc import imsave
import copy
#####################################################
"""
Configuration settings
"""
mean = np.array([128., 128., 128.], np.float32)
# Path to the textfiles for the trainings and validation set
train_file = './ColumbClustrTrain/ColmbClstrTrain1.txt'
AnnoBase = './BoundingBox/' # bounding box information of participants 
MaskPath = './MaskGenetn/'  # Mask storage path
MOMENTUM = 0.9
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
batch_size = 1 # number CAms to be taken at a time
#################################Excitation Map generator ########
train_generator = ImageDataGenerator2(train_file, shuffle=False, basePath='./MaskGenetn/SpkNonExTrain1/')  # speak and not speak CAMs taking 
num_classes = 2
Train_batches_per_epoch = np.ceil(train_generator.data_size / batch_size).astype(np.int16)

##########################Mask Generation #############
def Mask_Gen(Img):
    thresh_fxd = 0.20
    kernel = np.ones((11, 11), np.uint8)
    mask = np.zeros(shape=[512, 832], dtype=float)
    j =0
    fnlthresh = thresh_fxd + cv2.mean(Img[j])[0]
    Img2 = Img.astype('uint8')
    th2, im_bw = cv2.threshold(Img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = np.logical_or.reduce([mask, im_bw])
    mask_Fnl = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    return mask_Fnl

if __name__ == "__main__":

    temp = np.empty([0, 2048])
    label = np.empty([0, 1])
    batch_size = 1; 

    for _ in range(Train_batches_per_epoch ):
        batch_tx, batch_ty, AnnoLoctn,Paths,_ = train_generator.next_batch(batch_size)
        #print(Paths)
        test_count = 0
        MaskImg = []
        for j in range(batch_size):
            ##################Mask Label Image Generation###########
            labls = batch_ty[j]
            peopl = len(labls)
            #####################People Location###############
            Imheight= batch_tx[0].height# inp
            Imwidth= batch_tx[0].width# inp
            Locatns = AnnoLoctn[j]
            Person1 = np.multiply((labls[0]+1),np.ones(shape=[Imheight,Locatns[1]], dtype=int))
            Person2 = np.multiply((labls[1] + 1), np.ones(shape=[Imheight,(Locatns[3]-Locatns[2]-1)], dtype=int))
            All = np.hstack((Person1,Person2))
            if(peopl>2):
                Person3 = np.multiply((labls[2] + 1),
                                        np.ones(shape=[Imheight, (Locatns[5] - Locatns[4] - 1)], dtype=int))
                All = np.hstack((All, Person3))
            Fnl = resize(All, (Imheight,Imwidth),preserve_range=True)
            Fnl_Spk    = copy.copy(Fnl)
            Fnl_NonSpk = copy.copy(Fnl)
            Fnl_Spk[Fnl_Spk==1]    = 0
            Fnl_NonSpk[Fnl_NonSpk== 2]   = 0

            Spk_mskExc    = batch_tx[0,:,:]
            NonSpk_mskExc = batch_tx[1, :, :]
            Spk_msk       = Mask_Gen(np.multiply(Spk_mskExc, Fnl_Spk ),)
            NonSpk_msk    = Mask_Gen(np.multiply(NonSpk_mskExc, Fnl_NonSpk),)
            Fnl_Spk = np.multiply(Spk_msk,2)
            Fnl2    = Fnl_Spk + NonSpk_msk  #Fnl_MakGeneratd
            ##################################done###########################
            names = MaskPath +"/Mask/Img" +  str(test_count).zfill(5)  + '.jpg' 
            test_count = test_count+1
            
            Fnl2= np.multiply(Fnl2,127)
            imsave(names, Fnl2)
           