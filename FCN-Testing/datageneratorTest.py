import numpy as np
import cv2
import os
import re
from natsort import natsorted
AnnoBase = './BoundingBox/'
"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""

class ImageDataGeneratorTest:
    def __init__(self, class_list, horizontal_flip=False, shuffle=True,basePath = './PersonOptical1/', 
                 mean = np.array([128., 128., 128.],np.float32), scale_size=(1280,720),#(832,512)
                 nb_classes = 2): #[103.939, 116.779, 123.68],[128., 128., 128.]
        
                
        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.basePath =basePath
        self.scale_size = scale_size
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()


    def read_class_list(self,fileVAD,bbox_file,fullPath):
        """
        Scan the image file and get the image paths and labels
        """
        DI_Datapth = self.basePath
        jpglist = [file for file in os.listdir(DI_Datapth) if file.endswith('.jpg')]
        jpglist = natsorted(jpglist)
        #i = 0
        self.images = []
        self.Masks = []
        self.labels = []
        self.locatn = []
        self.FilName= []
    ############################
        FileVAD = fullPath+fileVAD
        FileBBOX = fullPath + bbox_file
        print('AnnotationFile', FileVAD)
        ####################Annotation processing
        f_Vad = open(FileVAD)
        f_BBOX = open(FileBBOX)
        lines_Vad = f_Vad.readlines()
        lines_Bbox = f_BBOX.readlines()
        ###############BBOX Loop
        VAD_len = len(lines_Vad)

        bbox_len = len(lines_Bbox)
        #lbl_VAD = (lines_Vad[i][:].split())[1]
        for i in range(2,(len(lines_Vad)-2),10):
            img_num = np.floor(int((lines_Vad[i][:].split())[0])/10) -37
            lbl_VAD = int((lines_Vad[i][:].split())[1])
            ####################VAD label##############
            for i in range(2, bbox_len - 2, 10):
                img_num2 = np.floor(int((lines_Bbox[i][:].split())[0]) / 10) - 37
                if(img_num==img_num2):
                    x11 = int((lines_Bbox[i][:].split())[1])
                    x12 = int((lines_Bbox[i][:].split())[2])
                    ############
                    x21 = int((lines_Bbox[i][:].split())[3])
                    x22 = int((lines_Bbox[i][:].split())[4])
                    Fnl_Anno = [x11, x12, x21, x22]
                    self.locatn.append(Fnl_Anno)

            imageNam = jpglist[int(img_num)]

            self.images.append(DI_Datapth + imageNam )
            self.labels.append(np.asanyarray(lbl_VAD, dtype=np.int16))

        self.data_size = len(self.images)
        print('the length of test files',(self.data_size))

        
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        #images = self.images.copy()
        images = self.images[:]
        #labels = self.labels.copy()
        labels = self.labels[:]
        self.images = []
        self.labels = []
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        paths = self.images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]

        ImgName = paths[0]
        self.pointer += batch_size
        
        anno   = np.ndarray([batch_size, 4],dtype=np.int16)

        for i in range(len(paths)):

            img = cv2.imread(paths[i],cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            str1= self.locatn[i]
            anno[i] = np.asanyarray(str1,dtype=np.int16)


        return img, anno,labels,ImgName

    def get_Labels(self):
        labels = self.labels
        return labels
    def get_FilName(self):
        label = self.FilName
        return label



