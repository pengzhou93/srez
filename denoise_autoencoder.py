import scipy
import time
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
tf.app.flags.DEFINE_integer('resume', False,
                            "Resume training.")

tf.app.flags.DEFINE_integer('summary_period', 200,
                            "Number of batches between summary data dumps")
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
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint/auto_encoder',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 10000,
                            "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('train_dir', 'train/auto_encoder',
                           "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_string('log_dir', 'log',
                           "Log folder where training logs are dumped.")

tf.app.flags.DEFINE_integer('train_time', 2000,
                            "Time in minutes to train the model")

tf.app.flags.DEFINE_string('dataset', 'dataset',
                           "Path to the dataset directory.")

tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_string('run', 'train',
                            "Which operation to run. [demo|train]")

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_integer('sample_size', 64,
                            "Image sample size in pixels. Range [64,128]")

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")


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

def _downscale(images, K):
    """Differentiable image downscaling by a factor of K"""
    arr = np.zeros([K, K, 3, 3])
    arr[:,:,0,0] = 1.0/(K*K)
    arr[:,:,1,1] = 1.0/(K*K)
    arr[:,:,2,2] = 1.0/(K*K)
    dowscale_weight = tf.constant(arr, dtype=tf.float32)
    
    downscaled = tf.nn.conv2d(images, dowscale_weight,
                              strides=[1, K, K, 1],
                              padding='SAME')
    return downscaled

def autoencoder_mode(sess, input_pl, channels = 3):
    """
     Create autoencoder mode
    
    Parameters:
    ----------
     sess : tf.Session
        
    Returns:
    ----------
     auto_en_output
         Output of autoencoder
     auto_en_var_list : list
         Trainable variables of autoencoder
    """
    with tf.variable_scope('auto_encoder') as scope:
        auto_en_output, auto_en_var_list = srez_model._generator_model(sess, input_pl, None, channels)
        K = int(auto_en_output.shape[1]) // int(input_pl.shape[1])
        assert K == 4, "downscale 4"
        downscaled = _downscale(auto_en_output, K)
    return [downscaled, auto_en_var_list]

def _get_summary_image(auto_en_input, auto_en_output, max_samples = 16):
    with tf.variable_scope('summary_image') as scope:
        clipped = tf.maximum(tf.minimum(auto_en_output, 1.0), 0.0)
        assert int(auto_en_output.shape[1]) == auto_en_input.shape[1]
        images = tf.concat(axis = 2, values = [clipped, auto_en_input])
        images = tf.concat(axis = 0, values = [images[i, :, :, :] for i in range(max_samples)])
    return images
def _summarize_progress(image, batch, suffix):
    filename = 'auto_en_batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))
def _save_checkpoint(train_data, batch):
    td = train_data

    newname = 'checkpoint_new.txt'
    newname = os.path.join(FLAGS.checkpoint_dir, newname)

    # Generate new checkpoint
    td.saver.save(td.sess, newname, global_step = batch)
    print("Checkpoint saved to %s, batch %d"%(newname, batch))


def train_model(train_data):
    td = train_data
    test_labels = td.sess.run(td.test_labels)
    images = _get_summary_image(test_labels, td.auto_en_output)

    summary_op = tf.summary.merge_all()

    if FLAGS.resume:
        ckpt_path = td.ckpt.model_checkpoint_path
        td.saver.restore(td.sess, ckpt_path)
        print('Resume from ', ckpt_path)
    else:
        init = tf.global_variables_initializer()
        td.sess.run(init)

    tf.get_default_graph().finalize()
    start_time  = time.time()
    done  = False
    batch = 0
    while not done:
        batch += 1
        auto_en_input = td.sess.run(td.auto_en_input)
        feed_dict = {td.auto_en_input_pl : auto_en_input, \
                     td.global_step_pl : batch}

        td.sess.run(td.auto_en_minimize, feed_dict)

        if batch % 50 == 0 or batch < 50:
            auto_en_l1_loss = td.sess.run(td.auto_en_l1_loss, feed_dict)
            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            print('Progress[%3d%%], ETA[%4dm], Batch [%4d], \
            auto_en_l1_loss[%3.3f]' % \
                  (int(100*elapsed/FLAGS.train_time), FLAGS.train_time - elapsed, batch, \
                   auto_en_l1_loss))

            # Finished?            
            current_progress = elapsed / FLAGS.train_time
            if current_progress >= 1.0:
                done = True
            # Summary
            merge = td.sess.run(summary_op, feed_dict = feed_dict)
            td.summary_writer.add_summary(merge, batch)
            td.summary_writer.flush()
           
        if batch % FLAGS.summary_period == 0:
            # Show progress with test features
            feed_dict = {td.auto_en_input_pl : test_labels}
            summary_image = td.sess.run(images, feed_dict=feed_dict)
            _summarize_progress(summary_image, batch, 'out')
           
        if batch % FLAGS.checkpoint_period == 0:
            # Save checkpoint
            _save_checkpoint(td, batch)

    _save_checkpoint(td, batch)
    print('Finished training!')
            

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
    _, train_labels = srez_input.setup_inputs(sess, train_filenames)
    _, test_labels  = srez_input.setup_inputs(sess, test_filenames)

    # Add some noise during training (think denoising autoencoders)
    noise_level = .03
    auto_en_input = train_labels + \
                           tf.random_normal(train_labels.get_shape(), stddev=noise_level)

    channels  = int(train_labels.get_shape()[3])
    rows = int(train_labels.get_shape()[1])
    cols = int(train_labels.get_shape()[2])
    # Placeholder for gene_input
    auto_en_input_pl = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, rows, cols, channels], \
                                 name = 'gene_input_pl')

    # Create and initialize model
    auto_en_output, auto_en_var_list = autoencoder_mode(sess, auto_en_input_pl, channels)

    assert int(auto_en_output.shape[1]) == rows
    # loss
    auto_en_l1_loss = tf.reduce_mean(tf.abs(auto_en_output - auto_en_input_pl), \
                                     name = 'auto_en_l1_loss')
    # Summary
    tf.summary.scalar('auto_en_l1_loss', auto_en_l1_loss)
    summary_auto_en_loss = tf.summary.FileWriter(FLAGS.log_dir + '/auto_en_loss')
    summary_auto_en_loss.add_graph(auto_en_l1_loss.graph)

    # Optimizer
    global_step_pl = tf.placeholder(dtype = tf.int64, name = 'global_step_pl')
    starter_learning_rate = FLAGS.learning_rate_start
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_pl,
                                           FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    tf.summary.scalar('learning_rate_pl', learning_rate)
    auto_en_opti = tf.train.RMSPropOptimizer(learning_rate = learning_rate, \
                                             name = 'auto_en_optimizer_RMS')
    auto_en_minimize = auto_en_opti.minimize(auto_en_l1_loss, \
                                             var_list = auto_en_var_list, \
                                             name = "auto_en_minimize")
    
    # Save model
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    saver = tf.train.Saver(max_to_keep = 2)

    # Train model
    train_data = TrainData(locals())
    train_model(train_data)

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
