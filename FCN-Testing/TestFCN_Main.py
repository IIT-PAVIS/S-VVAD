import tensorflow as tf
import numpy as np
import resnet_fcn
from datageneratorTest import ImageDataGeneratorTest  
import time
import  scipy
import  cv2
import os
from   scipy.io import savemat
from natsort import natsorted
"""
Configuration settings
"""
mean = np.array([128., 128., 128.], np.float32)
# Path to the textfiles for the trainings and validation set
train_file = './ColumbClustrTrain/ColmbClstrTrain2.txt'#ColmbClstrTrain1.txt'#ColmbClstrSngle.txt'#ColmbClstrSngle.txt'#ColmbClstrTrain1.txt'
val_file = './ColumbClustrTest/ColmbTest2.txt'  # Columb2
AnnoBase = './BoundingBox/'
##############  Reload the Trained FCN
PATH = './FCN_Traind/TrainSetforRealVADTdy520/Backup/'
META_FN = PATH + "model20.ckpt-20.meta"  # 
CHECKPOINT_FN = PATH + "model20.ckpt-20.meta"
CHECKPOINT_FN2 = PATH + "model20.ckpt-20"

MOMENTUM = 0.9
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './FCN_Traind2/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 1, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 1000, "max steps")#500000

##############################parameters
batch_size = 1#128
Num_Epoches = 1
basePath='./RealVADDataset/' #ModifiedColumbia
#################################Train Test read########
val_generator = ImageDataGeneratorTest(val_file, shuffle=False,
                                    basePath='./RealVADDataset/')  # ../PersonDynamic10/')#./PersonOptical/')

###################################Accuracy Measure##################
def Accuracy(lablP, true_labels):
    len_p = len(lablP)
    len_g = len(true_labels)
    if len_p<len_g :
        Labl_P = lablP
        Labl_G = true_labels[:len_p]
    else:
        Labl_P = lablP[:len_g]
        Labl_G = true_labels
    lablP = Labl_P
    true_labels = Labl_G
    TP = 0;    TN = 0;    FP = 0;    FN = 0
    TP = TP + np.sum(np.logical_and(lablP == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = TN + np.sum(np.logical_and(lablP == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = FP + np.sum(np.logical_and(lablP == 1, true_labels == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = FN + np.sum(np.logical_and(lablP == 0, true_labels == 1))
    # print('TP: %i' %(TP))
    print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP, FP, TN, FN))
    PRCN = np.float(TP)/np.float(TP+FP+0.000001)
    RCALL= np.float(TP) / np.float(TP + FN + 0.000001)
    F1Score = 2*PRCN*RCALL/(PRCN+RCALL+0.0000001)
    Accuracy = np.float(TP+TN)/np.float(TP+TN+FP+FN+0.000001)
    print('Percision: %f, Recall: %f, Fscore: %f,, Accuracy: %f'%(PRCN, RCALL, F1Score,Accuracy))

if __name__ == "__main__":
    xse_M = tf.placeholder(tf.float32, [None, None, None, 3], name='xse_M')
    step = 100
    #Match_Tx, maskImg = Backbone()
    saver = tf.train.import_meta_graph(CHECKPOINT_FN,input_map={"xse:0": xse_M})
    graph = tf.get_default_graph()
    ####changing the input  ####1280,720
    #xse_M = tf.placeholder(tf.float32, [None, 720, 1280, 3], name='xse_M')
    ##########get tensor by name####
    pred = graph.get_tensor_by_name("predct:0")
    xse = graph.get_tensor_by_name("xse_M:0")

    y = graph.get_tensor_by_name("labels:0")
    #pred = graph.get_tensor_by_name("predct:0")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)

    sess = tf.Session()

    saver.restore(sess, tf.train.latest_checkpoint(PATH))
    #_ = tf.import_graph_def(graph, input_map={"xse:0": xse_M})
    ######################if It is last Epoch Run the test##############

    imgcount = 37
    Gnd_labl = []
    Prd_labl = []
    fullPath = basePath + 'DynamicFrames/'#+'/Extr'
    filelist = [file for file in os.listdir(fullPath) if file.endswith('.jpg')]
    filelist = natsorted(filelist)
    mean = np.array([128., 128., 128.], float)
    scale_size = (1280,720)#(1480,860)#(2300,1260)#(1940,1080)#1280,720 #(832,512)#300:1132,124:636 #1664,963#832 #1940
    #scale_size2 = (1664, 963) #(1940,1080)
    for fileRGB in filelist:  # j>=0 :# == (Num_Epoches-1):
        imfullPath = fullPath+fileRGB
        imgfile = cv2.imread(imfullPath)
        img = imgfile.astype(np.float32)-mean#/127 #[124:(636-80),300:(1132-140)]
        img = cv2.resize(img, (scale_size[0], scale_size[1]))
        images = np.ndarray([1, scale_size[1], scale_size[0], 3])
        images[0] = img
        Pred_img = sess.run(pred, feed_dict={
            xse_M: images
        })

        ################image Writting
        for l in range(1): #len(Labl)
            Path = './GenMask/RealVAD/'+'Dyanmic_'+str(imgcount)+'.jpg'    #Img_' + str(imgcount) + '_' + str(Labl[l][0]) + '_' + str(Labl[l][1]) + '.jpg'
            PredIm = Pred_img[l, :, :, 0]
            PredIm=PredIm.astype(float)
            PredIm = st.resize(PredIm, (720,1280))
            Fnl = np.multiply(127, PredIm)
            Fnl = Fnl.astype(int)
            cv2.imwrite(Path, Fnl)  # save the generated Mask for Test Images
            imgcount = imgcount + 1


