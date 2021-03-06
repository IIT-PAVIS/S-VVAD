import numpy as np
import cv2
import os
import re
#import skimage as sk
from natsort import natsorted
AnnoBase = './BoundingBox/' #bounding Box file path
"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""
kernel = np.ones((9, 9), np.uint8) # morpholigical fileter kernel for noise removal
class ImageDataGeneratorTrain:
    def __init__(self, class_list, horizontal_flip=False, shuffle=True,basePath = './ColumbDynamic/', 
                 mean = np.array([128., 128., 128.],np.float32), scale_size=(560,340),# input image size
                 nb_classes = 2): #[103.939, 116.779, 123.68],[128., 128., 128.]
        
                
        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.basePath =basePath
        self.scale_size = scale_size
        self.pointer = 0
#       self.natural_sort(file)
        self.read_class_list(class_list)

        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self,class_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.Masks  = []
            self.labels = []
            self.locatn = []
            for l in lines:
                items = l.split()
                #############location of persons###########
                AnnoFile = AnnoBase + items[0] + '.txt';
                f = open(AnnoFile)
                lines = f.readlines()
                NPersons = len(lines)
                self.persons = NPersons-1;
                x11 = (lines[1][:].split())[1]
                x12 = (lines[1][:].split())[2]
                ############
                x21 = (lines[2][:].split())[1]
                x22 = (lines[2][:].split())[2]
                Fnl_Anno = [x11, x12, x21, x22]
                if(NPersons>3):
                    x31 = (lines[3][:].split())[1]
                    x32 = (lines[3][:].split())[2]
                    Fnl_Anno = [x11,x12,x21,x22,x31,x32]
                #############location of persons###########

                fullPath =  self.basePath+items[0]+'/'  #print(fullPath)
                filelist = [file for file in os.listdir(fullPath) if file.endswith('.jpg')]
                fullPathMsk = './FCNMaskAnntn/TrainSetFull'+fullPath[21:-1]+'/'                 #TrainSet1M
                filelistMsk = [file for file in os.listdir(fullPathMsk) if file.endswith('.jpg')]
                filelist = natsorted(filelist)
                filelistMsk = natsorted(filelistMsk)
                i = 0
                for file in filelist:
                    #print(file)
                    self.images.append(fullPath+file)
                    self.Masks.append(fullPathMsk + filelistMsk[i])
                    i = i+1
                    flsplit = file.split('_')
                    lenth = len(flsplit)
                    #print(lenth)
                    l1 = int(flsplit[3])
                    if(lenth>5):
                        l3 = int((flsplit[5]).split('.')[0])
                        l2 = int(flsplit[4])
                        fnl_Labl = [l1, l2, l3]
                    else:
                        l2 = int((flsplit[4]).split('.')[0])
                        fnl_Labl = [l1, l2]

                    self.labels.append(np.asanyarray(fnl_Labl, dtype=np.int16))#int(1))#file.split('_')[1]))
                    self.locatn.append(Fnl_Anno)
            
            #store total number of data
            self.data_size = len(self.labels)
            print('the Number of Training files',(self.data_size))
        
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        #images = self.images.copy()
        images = self.images[:]
        #labels = self.labels.copy()
        labels = self.labels[:]
        masks  = self.Masks[:]

        self.images = []
        self.labels = []
        self.Masks  = []
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])
            self.Masks.append(masks[i])
                
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
        pathsMsk = self.Masks[self.pointer:self.pointer + batch_size]
        #update pointer
        self.pointer += batch_size
        
        # Read images
        images = np.ndarray([batch_size, self.scale_size[1], self.scale_size[0], 3])
        masks = np.ndarray([batch_size, self.scale_size[1], self.scale_size[0]],dtype=np.int16)
        anno   = np.ndarray([batch_size, 3*2],dtype=np.int16)
        #print('the Length of batch is ',len(paths))
        for i in range(len(paths)):
            img = cv2.imread(paths[i])
            Mskimg = cv2.imread(pathsMsk[i])
            Msk = cv2.cvtColor(Mskimg, cv2.COLOR_BGR2GRAY)
            #Msk = cv2.erode(Msk.astype(np.uint8), kernel, iterations=2)
            #flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.20:
                img = cv2.flip(img, 1) # 1 for horizontal flip
                Msk = cv2.flip( Msk,1) 
            #if np.random.random() < 0.40:
                img2 = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
                msk = cv2.resize(Msk, (self.scale_size[0], self.scale_size[1]))


            else: 
                Msk2 = cv2.resize(Msk,(640,400)) #(555,360)#460,260#(560,340)
                img  = cv2.resize(img,(640,400))

                X_i = np.random.randint(0,80)
                Y_i = np.random.randint(0,60)
                #rescale image
                img2 = img[Y_i:(Y_i+340),X_i:(X_i+560),:]#cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
                msk = Msk2[Y_i:(Y_i+340),X_i:(X_i+560)]#cv2.resize(Msk2, (self.scale_size[0], self.scale_size[1]))
            imgF = img2.astype(np.float32)            
            #subtract mean
            imgF -= self.mean
                                                                    
            images[i] = np.float32(imgF)
            ####Quantization of Mask Values ###############
            msk[msk<=60] = 0
            msk[msk > 200] = 2
            msk[(msk > 60) & (msk <= 200)] = 1
            masks[i] = msk
            str1= self.locatn[i]
            if(len(str1)<5):
                str1 = str1+['0','0']
            anno[i] = np.asanyarray(str1,dtype=np.int16)

        # Expand labels to one hot encoding
        #one_hot_labels = np.zeros((batch_size, self.n_classes))
        #for i in range(len(labels)):
        #    one_hot_labels[i][labels[i]] = 1
        #labels =
        images=np.float32(images)
        #return array of images  labels and Annotation of localization
        return images, masks, anno#one_hot_labels#,paths


