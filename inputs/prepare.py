import numpy as np
import os
import scipy.misc
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def get_summary_image(feature, label, gene_output, max_samples=10):
    """Merge several test images in one big image for testing
    """
    with tf.variable_scope('summary_images') as scope:
        size = [label.shape[1], label.shape[2]]
        nearest = tf.image.resize_nearest_neighbor(feature, size)
        nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)
    
        bicubic = tf.image.resize_bicubic(feature, size)
        bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)
    
        clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)
    
        image   = tf.concat(axis = 2, values = [nearest, bicubic, clipped, label])
    
        image = image[0:max_samples,:,:,:]
        image = tf.concat(axis = 0, values = [image[i,:,:,:] for i in range(max_samples)])
    return image
    
def summarize_progress(image, imgs_dir, batch):
    filename = 'batch%08d.png' % (batch)
    filename = os.path.join(imgs_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("\t Saved %s" % (filename,))

def save_checkpoint(train_data, ckpt_dir, batch, ckpt_name = 'ckpt'):
    td = train_data
    ckpt_name = os.path.join(ckpt_dir, ckpt_name)
    td.saver.save(td.sess, ckpt_name, global_step = batch)
    print("\t Checkpoint saved to %s, batch %d"%(ckpt_name, batch))


class ScopeData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def prepare_dirs():
    # images dir
    imgs_dir = os.path.join(FLAGS.root_dir,
                            FLAGS.train_type,
                            'l1_{}'.format(FLAGS.gene_l1_factor), FLAGS.train_dir)
    if not tf.gfile.Exists(imgs_dir):
        tf.gfile.MakeDirs(imgs_dir)

    # Create checkpoint dir (do not delete anything)
    ckpt_dir = os.path.join(FLAGS.root_dir,
                            FLAGS.train_type,
                            'l1_{}'.format(FLAGS.gene_l1_factor), \
                            FLAGS.checkpoint_dir)
    if not tf.gfile.Exists(ckpt_dir):
        tf.gfile.MakeDirs(ckpt_dir)

    # log dir
    log_dir = os.path.join(FLAGS.root_dir,
                           FLAGS.train_type,
                           'l1_{}'.format(FLAGS.gene_l1_factor), \
                           FLAGS.log_dir)

    if tf.gfile.Exists(log_dir) and FLAGS.delete_log:
        tf.gfile.DeleteRecursively(log_dir)
        
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    dirs = ScopeData(locals())

    return dirs

# Prepare dirs for residual_gan_train.py
def prepare_res_gan_dirs():
    # images dir
    imgs_dir = os.path.join(FLAGS.root_dir,
                            FLAGS.train_type,
                            'l1_{}'.format(FLAGS.gene_l1_factor), FLAGS.train_dir)
    imgs_lf_dir = os.path.join(imgs_dir, 'lowf')
    if not tf.gfile.Exists(imgs_lf_dir):
        tf.gfile.MakeDirs(imgs_lf_dir)

    imgs_hf_dir = os.path.join(imgs_dir, 'highf')
    if not tf.gfile.Exists(imgs_hf_dir):
        tf.gfile.MakeDirs(imgs_hf_dir)
    
    # Create checkpoint dir (do not delete anything)
    ckpt_dir = os.path.join(FLAGS.root_dir,
                            FLAGS.train_type,
                            'l1_{}'.format(FLAGS.gene_l1_factor), \
                            FLAGS.checkpoint_dir)
    if not tf.gfile.Exists(ckpt_dir):
        tf.gfile.MakeDirs(ckpt_dir)

    # log dir
    log_dir = os.path.join(FLAGS.root_dir,
                           FLAGS.train_type,
                           'l1_{}'.format(FLAGS.gene_l1_factor), \
                           FLAGS.log_dir)

    if tf.gfile.Exists(log_dir) and FLAGS.delete_log:
        tf.gfile.DeleteRecursively(log_dir)
        
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

    dirs = ScopeData(locals())

    return dirs

def get_nn_bi_summary_image(feature, label):
    """Merge several test images in one big image for testing
    """
    with tf.variable_scope('summary_images') as scope:
        size = [label.shape[1], label.shape[2]]
        nearest = tf.image.resize_nearest_neighbor(feature, size)
        nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)
    
        bicubic = tf.image.resize_bicubic(feature, size)
        bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)
    
    return nearest, bicubic
