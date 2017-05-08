from inputs import srez_input

import tensorflow as tf
import os

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
    all_filenames = get_all_filenames(directory)
    train_filenames = all_filenames[:-FLAGS.test_vectors]
    test_filenames  = all_filenames[-FLAGS.test_vectors:]

    # Setup async input queues
    # train_features : down sample images(4x)   train_labels : original images(64 64 3)
    train_features, train_labels = srez_input.setup_inputs(sess, train_filenames)
    test_features,  test_labels  = srez_input.setup_inputs(sess, test_filenames)

    return [train_features, train_labels, test_features, test_labels]
