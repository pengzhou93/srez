import srez_demo
import srez_input
import srez_model
import srez_train

import os.path
import random
import numpy as np
import numpy.random

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)
tf.app.flags.DEFINE_integer('resume', True,
                            "Resume training.")

tf.app.flags.DEFINE_float('gene_l1_factor', 0.8,
                          "Multiplier for generator L1 loss term")
# Learning rate
tf.app.flags.DEFINE_float('learning_rate_start', 0.00005,
                          "Starting learning rate used for AdamOptimizer")

tf.app.flags.DEFINE_integer('decay_steps', 20000,
                            "Number of batches until learning rate is halved")

tf.app.flags.DEFINE_integer('decay_rate', 0.5,
                            "Decay learning rate")
# 
tf.app.flags.DEFINE_integer('test_vectors', 16,
                            """Number of features to use for testing""")
                            
tf.app.flags.DEFINE_integer('batch_size', 16,
                            "Number of samples per batch.")
# Checkpoint
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 10000,
                            "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('dataset', 'dataset',
                           "Path to the dataset directory.")

tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_string('run', 'train',
                            "Which operation to run. [demo|train]")

tf.app.flags.DEFINE_float('learning_beta1', 0.5,
                          "Beta1 parameter used for AdamOptimizer")

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_integer('sample_size', 64,
                            "Image sample size in pixels. Range [64,128]")

tf.app.flags.DEFINE_integer('summary_period', 200,
                            "Number of batches between summary data dumps")

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")

tf.app.flags.DEFINE_string('train_dir', 'train',
                           "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_string('log_dir', 'log',
                           "Log folder where training logs are dumped.")

tf.app.flags.DEFINE_integer('train_time', 2000,
                            "Time in minutes to train the model")

def show_images(images):
    """
     Show one batch images
    
    Parameters:
    ----------
     images : numpy
         shape : (num_images, height, width, channel)	  
     
    Returns:
    ----------
     None 
    """
    import cv2
    from scipy import misc
    num_images = images.shape[0]
    image = images[0]
    # image = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
    for i in range(num_images - 1):
        temp = images[i + 1]
        # temp = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2RGB)
        image = np.concatenate((image, temp), axis = 0)
    misc.imsave('test_images.png', image)
    # cv2.imshow("test images", image)
    # cv2.waitKey(0)
    assert 0, "Exit in show_images()"

def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)

    # Return names of training files
    if not tf.gfile.Exists(FLAGS.dataset) or \
       not tf.gfile.IsDirectory(FLAGS.dataset):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.dataset,))
  
    filenames = tf.gfile.ListDirectory(FLAGS.dataset)
    filenames = sorted(filenames)
#    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.dataset, f) for f in filenames]

    return filenames


def setup_tensorflow():
    # Create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    return sess, summary_writer

def _demo():
    # Load checkpoint
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=False)

    # Setup async input queues
    features, labels = srez_input.setup_inputs(sess, filenames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, features, labels)

    # Restore variables from checkpoint
    # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    saver.restore(sess, filename)

    # Execute demo
    srez_demo.demo1(sess)

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def _train():
    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    all_filenames = prepare_dirs(delete_train_dir=False)

    # Separate training and test sets
    train_filenames = all_filenames[:-FLAGS.test_vectors]
    test_filenames  = all_filenames[-FLAGS.test_vectors:]

    # TBD: Maybe download dataset here
    
    # Setup async input queues
    # train_features : down sample images(4x)   train_labels : original images(64 64 3)
    train_features, train_labels = srez_input.setup_inputs(sess, train_filenames)
    test_features,  test_labels  = srez_input.setup_inputs(sess, test_filenames)
    # XXX: debug
    test_images = sess.run(test_features)
    show_images(test_images)

    # Add some noise during training (think denoising autoencoders)
    noise_level = .03
    gene_input = train_features + \
                           tf.random_normal(train_features.get_shape(), stddev=noise_level)

    rows      = int(train_features.get_shape()[1])
    cols      = int(train_features.get_shape()[2])
    channels  = int(train_features.get_shape()[3])
    rows_label = int(train_labels.get_shape()[1])
    cols_label = int(train_labels.get_shape()[2])
    # Placeholder for train_features
    train_features_pl = tf.placeholder(tf.float32, \
                                       shape=[FLAGS.batch_size, rows, cols, channels], \
                                       name = 'train_features_pl')
     # Placeholder for gene_input
    gene_input_pl = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, rows, cols, channels], \
                                 name = 'gene_input_pl')
    # Placeholder for train_labels
    real_images_pl = tf.placeholder(tf.float32, \
                                 shape = [FLAGS.batch_size, rows_label, cols_label, channels], \
                                 name = 'real_images_pl')

    # Create and initialize model
    [gene_output, gene_var_list, \
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, gene_input_pl, real_images_pl)
    
    # Clip weight of discriminator
    with tf.variable_scope('d_clip') as _:
        d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in disc_var_list]

    # generator loss
    gene_loss = srez_model.create_generator_loss(disc_fake_output, gene_output, train_features_pl)
    # Summary
    tf.summary.scalar('gene_loss', gene_loss)
    summary_gene_loss = tf.summary.FileWriter(FLAGS.log_dir + '/gene_loss')
    summary_gene_loss.add_graph(gene_loss.graph)

    # discriminator loss
    disc_real_loss, disc_fake_loss = \
                     srez_model.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
    # Summary
    tf.summary.scalar('disc_real_loss', disc_real_loss)
    tf.summary.scalar('disc_fake_loss', disc_fake_loss)
    tf.summary.scalar('disc_loss', disc_loss)
    summary_disc_loss = tf.summary.FileWriter(FLAGS.log_dir + '/disc_loss')
    summary_disc_loss.add_graph(disc_loss.graph)
    
    # Optimizer
    # learning_rate_pl  = tf.placeholder(dtype=tf.float32, name='learning_rate')
    # global_step = tf.Variable(0, trainable=False)
    global_step_pl = tf.placeholder(dtype = tf.int64, name = 'global_step_pl')
    starter_learning_rate = FLAGS.learning_rate_start
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_pl,
                                           FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    tf.summary.scalar('learning_rate_pl', learning_rate)
    (gene_minimize, disc_minimize) = \
            srez_model.create_optimizers(gene_loss, gene_var_list,
                                         disc_loss, disc_var_list,
                                         learning_rate)

    # Save model
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    saver = tf.train.Saver(max_to_keep = 2)

    # Train model
    train_data = TrainData(locals())
    srez_train.train_model(train_data)

def main(argv=None):
    # Training or showing off?
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    if FLAGS.run == 'demo':
        _demo()
    elif FLAGS.run == 'train':
        _train()

if __name__ == '__main__':
  tf.app.run()
