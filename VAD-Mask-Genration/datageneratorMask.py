import numpy as np
import cv2
import os
import re
import skimage as sk
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


class ImageDataGeneratorMask:
    def __init__(self, class_list, horizontal_flip=False, shuffle=True, basePath='./GroupDynamic/',
                 mean=np.array([128., 128., 128.], np.float32), scale_size=(832,512),#(1280,720),#(832, 512),(1024,576) 560= 96 f1score
                 nb_classes=2):  

        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.basePath = basePath  
        self.scale_size = scale_size
        self.pointer = 0
        #        self.natural_sort(file)
        self.read_class_list(class_list)

        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self, class_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            self.locatn = []
            self.images_PPlDtct = []
            for l in lines:
                items = l.split()
                #############location of persons###########
                AnnoFile = AnnoBase + items[0] + '.txt';
                f = open(AnnoFile)
                lines = f.readlines()
                NPersons = len(lines)
                self.persons = NPersons - 1;
                x11 = (lines[1][:].split())[1]
                x12 = (lines[1][:].split())[2]
                ############
                x21 = (lines[2][:].split())[1]
                x22 = (lines[2][:].split())[2]
                Fnl_Anno = [x11, x12, x21, x22]
                if (NPersons > 3):
                    x31 = (lines[3][:].split())[1]
                    x32 = (lines[3][:].split())[2]
                    Fnl_Anno = [x11, x12, x21, x22, x31, x32]
                # Fnl_Anno = [x11, x12, x21, x22]
                #############location of persons###########
                humanDtctFullPath =  './BodyClstrBW/'+items[0]+'/'
                filelistBdClst = [humanDtctFullPath for humanDtctFullPath in os.listdir((humanDtctFullPath)) if humanDtctFullPath.endswith('.jpg')]
                filelistBdClst = natsorted(filelistBdClst)

                fullPath = self.basePath + items[0] + '/'+'Spk/'
                filelistSpk = [file for file in os.listdir((fullPath)) if file.endswith('.jpg')]
                #filelistNonSpk = [file for file in os.listdir((fullPath+'NonSpk/'))  if file.endswith('.jpg')]
                filelistSpk = natsorted(filelistSpk)
                i = 0
                for file in filelistSpk:
                    # print(file)
                    self.images_PPlDtct.append(humanDtctFullPath + filelistBdClst[i])
                    i = i+1
                    self.images.append(fullPath + file)
                    flsplit = file.split('_')
                    lenth = len(flsplit)
                    # print(lenth)
                    l1 = int(flsplit[3])
                    # l2 = int(flsplit[4])
                    # fnl_Labl = [l1, l2]
                    if (lenth > 5):
                        l3 = int((flsplit[5]).split('.')[0])
                        l2 = int(flsplit[4])
                        fnl_Labl = [l1, l2, l3]
                    else:
                        l2 = int((flsplit[4]).split('.')[0])
                        fnl_Labl = [l1, l2]

                    self.labels.append(np.asanyarray(fnl_Labl, dtype=np.int16))  # int(1))#file.split('_')[1]))
                    self.locatn.append(Fnl_Anno)

            # store total number of data
            self.data_size = len(self.labels)
            print('the length of test files', (self.data_size))

    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        # images = self.images.copy()
        images = self.images[:]
        # labels = self.labels.copy()
        labels = self.labels[:]
        self.images = []
        self.labels = []

        # create list of permutated index and shuffle data accoding to list
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
        paths_ppl = self.images_PPlDtct[self.pointer:self.pointer + batch_size]
        # update pointer
        self.pointer += batch_size

        # Read images
        images = np.ndarray([batch_size+1, self.scale_size[1], self.scale_size[0]],dtype=np.int8)
        anno = np.ndarray([batch_size, 3 * 2], dtype=np.int16)
        for i in range(len(paths)):#for i in rangelen(paths)):    #Modified For Loop
            str1 = self.locatn[i]
            length = len(str1)
            if(length>5):
                mainPath = paths[0][0:(len(paths[0])-29)]#32
                fileName = paths[0][(len(paths[0])-25):] #24
            else:
                mainPath = paths[0][0:(len(paths[0]) - 27)]  # 32
                fileName = paths[0][(len(paths[0]) - 23):]  # 24

            SpkFile = paths[i]
            NonSpkFile = mainPath + 'NonSpk/' + fileName
            imgSpk = cv2.imread(SpkFile,0)
            imgNonSpk = cv2.imread(NonSpkFile,0)
            PPlDetectImg = cv2.imread(paths_ppl[i],0)
            PPlDetectImg = cv2.resize(PPlDetectImg, (self.scale_size[0], self.scale_size[1]))

            # flip image at random if flag is selected
            #if self.horizontal_flip and np.random.random() < 0.5:
            #    img = cv2.flip(img, 1)

            # rescale image
            #img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            #img = img.astype(np.float32)

            # subtract mean
            #img -= self.mean

            images[i] = imgSpk#np.float32(img)
            images[i+1] = imgNonSpk  #

            if (len(str1) < 5):
                str1 = str1 + ['0', '0']
            anno[i] = np.asanyarray(str1, dtype=np.int16)

        # Expand labels to one hot encoding
        # one_hot_labels = np.zeros((batch_size, self.n_classes))
        # for i in range(len(labels)):
        #    one_hot_labels[i][labels[i]] = 1
        # labels =
        #images = np.float32(images)
        # return array of images and labels
        return images, labels, anno,paths,PPlDetectImg  # one_hot_labels#,paths


