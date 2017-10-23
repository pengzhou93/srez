from inputs import srez_input
import glob,re
import tensorflow as tf
import os
import numpy as np
import scipy.io
import scipy

from networks import srez_model
import  pywt
DATA_PATH = "291"
FLAGS = tf.app.flags.FLAGS

def get_all_filenames(directory):
    # Return names of training files
    if not tf.gfile.Exists(directory) or \
       not tf.gfile.IsDirectory(directory):
        raise FileNotFoundError("Could not find folder `%s'" % (directory,))

    filenames = tf.gfile.ListDirectory(directory)
    filenames = sorted(filenames)
#    random.shuffle(filenames)
    filenames = [os.path.join(directory, f) for f in filenames]

    return filenames

def get_batch_inputs(sess, directory):
    # Separate training and test sets
    all_filenames = glob.glob(os.path.join(directory, "*.JPEG"))
    #all_filenames = get_all_filenames(directory)
    #train_filenames = all_filenames[:-FLAGS.test_vectors]
    #test_filenames  = all_filenames[-FLAGS.test_vectors:]

    # Setup async input queues
    # train_features : down sample images(4x)   train_labels : original images(64 64 3)
    [train_features, train_labels] = srez_input.setup_inputs(sess, all_filenames)
    #test_features,  test_labels  = srez_input.setup_test_inputs(sess, test_filenames)

    return train_features,  train_labels

def get_train_list(data_path):
    l = glob.glob(os.path.join(data_path,"*"))
    #l = [f for f in l if re.search("^\d+.jpg$", os.path.basename(f))]
    train_list = []
    for f in l:
        #if os.path.exists(f):
            #if os.path.exists(f[:-4]+".jpg"): train_list.append([f, f[:-4]+".mat"])
        train_list.append(f)
    return train_list

def get_image_batch(train_list,batch_size):
    target_list = train_list
    input_list = []
    gt_list = []
    cbcr_list = []
    for pair in target_list:
        input_img = scipy.io.loadmat(pair[1])
        input_img = tf.reshape(input_img, [FLAGS.sample_size//4, FLAGS.sample_size//4, 3])
        input_img = tf.decode_raw(input_img, tf.uint8)
        #input_img=tf.image.resize_area(input_img,[FLAGS.sample_size//4, FLAGS.sample_size//4])
        gt_img = scipy.io.loadmat(pair[0])
        gt_img = tf.reshape(gt_img, [FLAGS.sample_size, FLAGS.sample_size, 3])
        gt_img = tf.decode_raw(gt_img, tf.uint8)
        #gt_img= tf.image.resize_area(gt_img,[FLAGS.sample_size, FLAGS.sample_size])
        input_list.append(input_img)
        gt_list.append(gt_img)
    input_list = np.array(input_list)
    gt_list = np.array(gt_list)
    features, labels = tf.train.batch([input_list, gt_list],
                                          batch_size=FLAGS.batch_size,
                                          num_threads=2,
                                          capacity = 3*FLAGS.batch_size,
                                          name='labels_and_features')
    return features, labels

def get_test_image(test_list, offset, batch_size):
    target_list = test_list[offset:offset+batch_size]
    input_list = []
    gt_list = []
    for pair in target_list:
        mat_dict = scipy.io.loadmat(pair[1])
        input_img = None
        if mat_dict.has_key("img_2"):   input_img = mat_dict["img_2"]
        elif mat_dict.has_key("img_3"): input_img = mat_dict["img_3"]
        elif mat_dict.has_key("img_4"): input_img = mat_dict["img_4"]
        else: continue
        gt_img = scipy.io.loadmat(pair[0])['img_raw']
        input_list.append(input_img[:,:,0])
        gt_list.append(gt_img[:,:,0])
    return input_list, gt_list

def get_test_input(sess,directory,dpdirectory):
    all_filenames = get_all_filenames(directory)
    #reader = tf.WholeFileReader()
    #filename_queue = tf.train.string_input_producer(all_filenames, shuffle = False)
    value = tf.read_file(all_filenames[12])
    #reader = tf.WholeFileReader()
    #filename_queue = tf.train.string_input_producer(all_filenames)
    #key, value = reader.read(filename_queue)

    channels = 3
    image = tf.image.decode_jpeg(value, name="dataset_image")
    #image.set_shape([None, None, channels])
    image = tf.image.rgb_to_grayscale(image)
    #image=tf.reshape(image,[1,None, None, channels])
    image = tf.cast(image, tf.float32)/255.0
    #[rows, cols, channels] = image.shape
    #image.set_shape([1,None, None, 1])
    image = tf.expand_dims(image,0)
    #down_filenames=get_all_filenames(dpdirectory)
    downsampled= srez_model._downscale(image,4)
    #reader = tf.WholeFileReader()
    #filename_queue = tf.train.string_input_producer(down_filenames)
    #dpkey, dpvalue = reader.read(filename_queue)
    #features = tf.image.decode_jpeg(downsampled, channels=channels, name="dataset_image")
    #features = downsampled[0,:,:,:]
    #image = image[0,:,:,:]
    #features=tf. reshape(features,[1,None, None, channels])
    #features = tf.cast(features, tf.float32) / 255.0
    #features, labels = tf.train.batch([downsampled, image],
                                      #batch_size=1,
                                      #num_threads=2,
                                      #capacity=1 * FLAGS.batch_size,
                                      #allow_smaller_final_batch=True,
                                      #name='images_and_features')
    #coord = tf.train.Coordinator()
    #tf.train.start_queue_runners(sess=sess, coord=coord)
    return downsampled, image

def get_test_input_v1(sess,directory):
    all_filenames = get_all_filenames(directory)
    test_features,  test_labels  = srez_input.setup_test_inputs_v1(sess, all_filenames)

    return test_features,  test_labels


def get_train_batch(train_list,batch_size):
    filenames = train_list
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    channels = 3
    image_size=64
    image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")
    image = tf.cast(image, tf.float32)/255.0
    image = tf.reshape(image, [1,image_size, image_size,3])
    downscaled = tf.image.resize_area(image, [image_size//4, image_size//4])
    downscaled = tf.reshape(downscaled, [image_size//4, image_size//4,3])
    image = tf.reshape(image, [image_size, image_size,3])
    features, labels = tf.train.batch([downscaled, image],
                                          batch_size=FLAGS.batch_size,
                                          num_threads=2,
                                          capacity = 3*FLAGS.batch_size,
                                          name='labels_and_features')
    return features,labels