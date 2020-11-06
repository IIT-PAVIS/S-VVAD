import tensorflow as tf
import numpy as np
import resnet_fcn
from datageneratorRealVAD import ImageDataGeneratorTrain  #datageneratorTest
from datageneratorTest import ImageDataGeneratorTest  #datageneratorTest
import time
import  scipy
import  cv2
#import matplotlib.pyplot as plt
import os
#from skimage.transform import resize
#import skimage as sk
#import skimage.measure as measure

#from scipy.misc import imsave
#####################################################
"""
Configuration settings
"""
mean = np.array([128., 128., 128.], np.float32)
# Path to the textfiles for the trainings and validation set
train_file = './ColumbClustrTrain/ColmbClstrTrain1.txt'#ColmbClstrTrain1.txt'#ColmbClstrSngle.txt'#ColmbClstrSngle.txt'#ColmbClstrTrain1.txt'
val_file = './ColumbClustrTest/ColmbTest1.txt'  # Columb2
AnnoBase = './BoundingBox/'
#######################################

PATH = './TraindTestVarbl224_356/Test1Varbl10Ep/'#"./CAM_Model/Clusters/Final/CS20E/"#C16/" './TraindTestVarbl224_356/Test1Varbl10Ep/'
META_FN = PATH + "model_epoch10.meta"  # for non/speak, "./CAM_Model/Pavis_Traind/TraindModelCluster/checkpointC1/"
CHECKPOINT_FN = PATH + "model_epoch10.ckpt.meta"
CHECKPOINT_FN2 = PATH + "model_epoch10.ckpt"

MOMENTUM = 0.9
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './FCN_Traind/TrainSet1111/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.5e-4, "learning rate.") #1e-5#1e-4
tf.app.flags.DEFINE_integer('batch_size', 4, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 1000, "max steps")#500000

##############################parameters
batch_size = 4#128
Num_Epoches = 20
# saver = tf.train.import_meta_graph(PATH + 'model_epoch1.ckpt.meta')
#################################Train Test read########
train_generator = ImageDataGenerator2(train_file, horizontal_flip=True, shuffle=True,
                                      basePath='./ColumbClustrDynamic/')  # ../PersonDynamic10/')#./PersonOptical/')
val_generator = ImageDataGeneratorTest(val_file, shuffle=False,
                                    basePath='./ColumbClustrDynamic/')  # ../PersonDynamic10/')#./PersonOptical/')

num_classes = 3
Train_batches_per_epoch = np.ceil(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.ceil(val_generator.data_size / batch_size).astype(np.int16)


if __name__ == "__main__":


#####################FCN Place Houlders###################
    X = tf.placeholder(tf.float32, [None, 512, 832, 3],name='X') #batch_size
    labels = tf.placeholder(tf.int32, [None, 512, 832],name='labels') #batch_size
    #train_X = np.random.randint(0, 256, size=[batch_size, 512, 832, 3]).astype(np.float)
    #train_y = np.random.randint(0, num_classes, [batch_size, 512, 832]).astype(np.int)###[3, 4, 6, 3]
    pred, logits = resnet_fcn.inference(xse, is_training=True, num_classes=num_classes, num_blocks=[3, 4, 6, 3])

    ###################
    b = tf.constant(1,dtype=tf.int64)#,shape=[128,7,7,2048])
    c = tf.constant(1,dtype=tf.float32)
    predictn  = tf.multiply(b,pred,name = "predct")
    logit     = tf.multiply(c,logits,name='logit')
    #predictn    =  (pred, name=''

    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels, name = "entropy")))
   
    saver = tf.train.Saver([var for var in tf.global_variables() if "scale_fcn" not in var.name])

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)

    sess = tf.Session()

    # Init global variables
    sess.run(tf.global_variables_initializer())

    # Restore variables
    restore_variables = True
    if restore_variables:
        saver.restore(sess, CHECKPOINT_FN2)

    # new saver
    saver = tf.train.Saver(tf.global_variables())
    for j in range(Num_Epoches): #Number of Epoches Loop
        for _ in range(Train_batches_per_epoch):
            start_time = time.time()

            step = sess.run(global_step)

            run_op = [train_op, loss]
            train_X, train_y, AnnoLoctn = train_generator.next_batch(batch_size)
            o = sess.run(run_op, feed_dict = {
                X:train_X,
                labels:train_y
                })
            loss_value = o[1]

            duration = time.time() - start_time

            if step % 5 == 0: # After every n batches log the loss values
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                'sec/batch)')
                print(format_str % (step, loss_value, examples_per_sec, duration))

        ######################Perform Test images mask generation on after t number of steps##############
        
            if step % 200 == 0:#j>=0 :# == (Num_Epoches-1):
                imgcount = 0
                Gnd_labl = []
                Prd_labl = []
                val_generator.reset_pointer()
                for _ in range(val_batches_per_epoch):
                    test_X, AnnoLoctn,Labl = val_generator.next_batch(batch_size)
                    Pred_img = sess.run(predictn , feed_dict={
                        X: test_X
                    })            

                    ################Segmentation mask Writting##############
                    #for l in range(len(Labl)):
                    #
                    #    Path = './GenMask/Img_'+ str(imgcount)+'_'+str(Labl[l][0])+'_'+str(Labl[l][1])+ '.jpg'
                    #    PredIm = Pred_img[l,:,:,0]
                    #    Fnl = np.multiply(127,PredIm)
                    #    cv2.imwrite(Path,Fnl)
                    #    imgcount = imgcount+1
                Performance(Prd_labl, Gnd_labl)
        ################################################
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model%d.ckpt' % step)
                saver.save(sess, checkpoint_path, global_step=global_step)
                #train_generator.shuffle_data()
                train_generator.reset_pointer()
                print('The Epoch Number is= ',j)

