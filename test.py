from inputs import celebA
from inputs import utils
from networks import srez_model
from networks import residual_gan_model
from networks import srcnn_model
import pywt
from PIL import Image
import scipy.misc
from scipy import misc
import numpy as np
import os
import pprint
import time
import tensorflow as tf
import math
from train import residual_gan_train
#from networks import utils


FLAGS = tf.app.flags.FLAGS
DATA_PATH = "subimage291"
TEST_DATA_PATH = "Set14_train"
DP_TEST_PATH ='test'

def splitImage(downsampled, image):
    downsampled2 = []
    image2 = []
    for c in range(2):
        for r in range(2):
            downsampled3 = downsampled[:,c * downsampled.shape[1]//2 : (c+1)*downsampled.shape[1]//2, r * downsampled.shape[2]//2 : (r+1) * downsampled.shape[2]//2,:]
            image3 = image[:,c * image.shape[1]//2 : (c+1) * image.shape[1]//2, r* image.shape[2]//2 : (r + 1) * image.shape[2]//2,:]
            downsampled2.append((downsampled3))
            image2.append((image3))
    return downsampled2,image2
def PSNR(hr,original):
    if hr.shape != original.shape:
        x = min(hr.shape[0],original.shape[0])
        y = min(hr.shape[1],original.shape[1])
        original = original[0:x,0:y]
        hr = hr[0:x+1,0:y+1]
    mse = np.mean((hr - original) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse))
def TEST(ckpt_path, data_path, sess):
    all_filenames = celebA.get_all_filenames(data_path)
    # value = tf.read_file(all_filenames)
    td = residual_gan_train.train_model()
    psnr=[]
    td.saver.restore(sess, ckpt_path)
    downsampled_sym, image_sym = celebA.get_test_input_v1(sess, data_path)
    # image = tf.image.decode_jpeg(value[i])
    # image = tf.image.rgb_to_grayscale(image)
    # image = tf.cast(image, tf.float32) / 255.0
    # image = tf.expand_dims(image, 0)
    # downsampled = srez_model._downscale(image, 4)

    for i in range(len(all_filenames)):
        # image = scipy.misc.imread(all_filenames[i],mode = 'L')
        # image = image[np.newaxis, :, :, np.newaxis]
        downsampled, image = sess.run([downsampled_sym, image_sym])
        downsampled = downsampled[np.newaxis,:]
        image = image[np.newaxis, :]
        image_test=image
        #downsampled, image = splitImage(downsampled,image)
        a=[]
        #for j in range(4):
            #downsampled2 = downsampled[j]
            #image2 = image[j]
        test_cA, test_cD = pywt.dwt2(downsampled, 'haar', axes=[-3, -2])
        test_CA, test_CD = pywt.dwt2(image,'haar', axes=[-3, -2])
        test_CD = np.concatenate(test_CD,axis=3)
        test_cD = np.concatenate(test_cD, axis=3)
        # test_cD = misc.imresize(test_cD,[16,64,64,3],'bicubic')
        # test_srcnn_output = (test_srcnn_output[:, :, :, 0, None], test_srcnn_output[:, :, :, 1, None], test_srcnn_output[:, :, :, 2, None])
        # test_subpixel_restore = pywt.idwt2((test_cA,test_srcnn_output),'haar',mode='sym',axes=[-3,-2])
        sum_feed_dict = {td.train_features_pl : downsampled,
                        td.real_images_pl : test_CA,
                        td.wavelet_coeff_features_pl : test_cD,
                        td.wavelet_coeff_labels_pl:test_CD}
        # generated super resolution images
        test_gene_LL,test_gene_wavelet = sess.run([td.gene_LL,td.gene_wavelet],
                                    feed_dict=sum_feed_dict)
        test_gene_wavelet = (test_gene_wavelet[:, :, :, 0, None], test_gene_wavelet[:, :, :, 1, None], test_gene_wavelet[:, :, :, 2, None])
        test_gene_output = pywt.idwt2((test_gene_LL, test_gene_wavelet), 'haar', mode='sym', axes=[-3, -2])
        test_gene_output = np.clip(test_gene_output, 0, 1)

        # residual images
            #test_residual_images = test_gene_output_images - test_subpixel_images
            #test_residual_images = utils.inverse_transform(test_residual_images)
        #a.append(test_gene_output_images)
        test_file = os.path.join('test_set14', 'subpixel_%d.png'%i)
        #b=np.concatenate((a[0],a[1]),axis=2)
        #d=np.concatenate((a[2],a[3]),axis=2)
        #e=np.concatenate((b,d),axis=1)
        f=test_gene_output[0,:,:,0]
        image_test=image_test[0,:,:,0]
        scipy.misc.imsave(test_file,f)
        print(PSNR(f,image_test))
        psnr.append(PSNR(f,image_test))
    print(np.mean(psnr))
if __name__ == '__main__':
    with tf.Session() as sess:
        if not os.path.exists('test_set14'):
            os.mkdir('test_set14')
        ckpt_path = "./results/shortcut_gen/l1_0.99/checkpoint/ckpt-83600"
        TEST(ckpt_path, TEST_DATA_PATH, sess)



