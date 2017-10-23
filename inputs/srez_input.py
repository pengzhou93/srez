import tensorflow as tf
import pywt
from networks import srez_model
FLAGS = tf.app.flags.FLAGS

def setup_inputs(sess, filenames, image_size=None, capacity_factor=3):
    with tf.variable_scope('setup_inputs') as scope:
        if image_size is None:
            image_size = FLAGS.sample_size
        
        # Read each JPEG file
        reader = tf.WholeFileReader()
        filename_queue = tf.train.string_input_producer(filenames)
        key, value = reader.read(filename_queue)
        channels = 3
        image = tf.image.decode_jpeg(value)
        # image = tf.image.decode_jpeg(value, name="dataset_image")
        #if image.shape[2] != 3:
            #image = tf.image.grayscale_to_rgb(image)
        # image.set_shape([None, None, channels])
    
        # Crop and other random augmentations
        #image = tf.image.random_flip_left_right(image)
        #image = tf.image.random_saturation(image, .95, 1.05)
        #image = tf.image.random_brightness(image, .05)
        #image = tf.image.random_contrast(image, .95, 1.05)
    
        #wiggle = 8
        #off_x, off_y = 25-wiggle, 25-wiggle
        crop_size = 64
        #crop_size_plus = crop_size + 2*wiggle
        #if image.shape[1] < 256 and image.shape[2] < 256:
        #image = tf.image.resize_images(image, [crop_size, crop_size], method=tf.image.ResizeMethod.BICUBIC, align_corners=False)

        #image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
        #image = tf.random_crop(image, [crop_size, crop_size, 3])
        # image = tf.image.resize_area(image, [crop_size, crop_size])
        image = tf.image.rgb_to_grayscale(image)
        image = tf.random_crop(image,[crop_size,crop_size,1])
        image = tf.reshape(image, [1, crop_size, crop_size, 1])
        image = tf.cast(image, tf.float32)/255.0

    
        # The feature is simply a Kx downscaled version
        K = 4
        #downsampled = tf.image.resize_area(image, [crop_size//K, crop_size//K])
        downsampled = srez_model._downscale(image, 4)
        #feature = tf.reshape(downsampled, [image_size//K, image_size//K, 3])
        #label   = tf.reshape(image,       [image_size,   image_size,     3])
        #feature = tf.reshape(downsampled,[crop_size//K, crop_size//K,1])
        feature = downsampled[0,:,:,:]
        #label = tf.reshape(image,[crop_size, crop_size,1])
        label = image[0,:,:,:]
        #image2 = tf.reshape(image,[crop_size,crop_size])
        # Using asynchronous queues
        features, labels = tf.train.batch([feature, label],
                                          batch_size=FLAGS.batch_size,
                                          num_threads=2,
                                          capacity = capacity_factor*FLAGS.batch_size,
                                          name='labels_and_features',
                                          allow_smaller_final_batch = False)

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord = coord)
      
    return features, labels

def setup_test_inputs(sess, filenames, image_size=None, capacity_factor=3):
    with tf.variable_scope('setup_test_inputs') as scope:
        if image_size is None:
            image_size = FLAGS.sample_size
        
        # Read each JPEG file
        reader = tf.WholeFileReader()
        filename_queue = tf.train.string_input_producer(filenames, shuffle = False)
        key, value = reader.read(filename_queue)
        channels = 3
        image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")
        image.set_shape([None, None, channels])
    
        # wiggle = 8
        # off_x, off_y = 25-wiggle, 60-wiggle
        # crop_size = 128
        # crop_size_plus = crop_size + 2*wiggle
        # image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
        # image = tf.random_crop(image, [crop_size, crop_size, 3])

        off_x, off_y = 25, 60
        crop_size = 128
        image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size, crop_size)
        image = tf.reshape(image, [1, crop_size, crop_size, 3])

        image = tf.cast(image, tf.float32)/255.0
    
        if crop_size != image_size:
            image = tf.image.resize_area(image, [image_size, image_size])
    
        # The feature is simply a Kx downscaled version
        K = 4
        downsampled = tf.image.resize_area(image, [image_size//K, image_size//K])
    
        feature = tf.reshape(downsampled, [image_size//K, image_size//K, 3])
        label   = tf.reshape(image,       [image_size,   image_size,     3])
    
        # Using asynchronous queues
        features, labels = tf.train.batch([feature, label],
                                          batch_size=FLAGS.batch_size,
                                          num_threads=2,
                                          capacity = capacity_factor*FLAGS.batch_size,
                                          name='labels_and_features')

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord = coord)
      
    return features, labels


def setup_test_inputs_v1(sess, filenames, image_size=None, capacity_factor=3):
    with tf.variable_scope('setup_test_inputs') as scope:
        if image_size is None:
            image_size = FLAGS.sample_size

        # Read each JPEG file
        reader = tf.WholeFileReader()
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
        key, value = reader.read(filename_queue)
        channels = 3
        image = tf.image.decode_jpeg(value,  name="dataset_image")
        image = tf.image.rgb_to_grayscale(image)
        # image.set_shape([None, None, channels])
        #shape = tf.shape(image)
        #image_height, image_width = shape[0], shape[1]

        # The feature is simply a Kx downscaled version
        image = tf.expand_dims(image, 0)
        K = 4
        #down_shape = tf.stack([image_height/K, image_width/K])
        #down_shape = tf.cast(down_shape, tf.int32)
        #downsampled = tf.image.resize_area(image, down_shape)
        image =tf.cast(image,tf.float32)/255.0
        downsampled = srez_model._downscale(image, 4)
        downsampled = tf.squeeze(downsampled, [0])
        # feature = tf.reshape(downsampled, [image_height // K, image_width // K, 3])
        label = tf.squeeze(image, [0])
        #label = tf.cast(label, tf.float32) / 255.0
        #downsampled = tf.cast(downsampled, tf.float32) / 255.0

        # Using asynchronous queues
        # features, labels = tf.train.batch([downsampled, label],
        #                                   batch_size=1,
        #                                   num_threads=1,
        #                                   capacity=capacity_factor * FLAGS.batch_size,
        #                                   name='labels_and_features')

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)

    return downsampled, label
