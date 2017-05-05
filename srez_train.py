import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS

def _get_summary_image(feature, label, gene_output, max_samples=10):
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
    
def _summarize_progress(image, imgs_dir, batch):
    filename = 'batch%06d.png' % (batch)
    filename = os.path.join(imgs_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))

def _save_checkpoint(train_data, ckpt_dir, batch):
    td = train_data

    newname = 'checkpoint_new.txt'
    newname = os.path.join(ckpt_dir, newname)

    # Generate new checkpoint
    td.saver.save(td.sess, newname, global_step = batch)
    print("Checkpoint saved to %s, batch %d"%(newname, batch))


def train_model_wgan(train_data):
    td = train_data
    dirs = td.dirs
    # Cache test features and labels (they are small)
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])
    num_samples = FLAGS.test_vectors
    image = _get_summary_image(test_feature, test_label, td.gene_output, num_samples)

    summarie_op = tf.summary.merge_all()

    batch = 0
    if FLAGS.resume:
        ckpt_path = td.ckpt.model_checkpoint_path
        td.saver.restore(td.sess, ckpt_path)
        batch = int(ckpt_path.split('-')[-1])
        print('Resume from ', ckpt_path)
    else:
        # tensorflow version 1.0
        init = tf.global_variables_initializer()
        td.sess.run(init)
        print('Training from scratch!')
        
    tf.get_default_graph().finalize()
    start_time  = time.time()
    done  = False
    while not done:
        batch += 1
        gene_loss = disc_real_loss = disc_fake_loss = -1.234
        
        train_features, gene_input, real_images = td.sess.run([td.train_features, \
                                                               td.gene_input, \
                                                               td.train_labels])
        feed_dict = {td.train_features_pl : train_features, \
                     td.gene_input_pl : gene_input, \
                     td.real_images_pl : real_images, \
                     td.global_step_pl : batch}

        # Update discriptor
        d_iters = 5
        # if batch % 500 == 0 or batch < 25:
        #     d_iters = 100
        for _ in range(0, d_iters):
            td.sess.run(td.d_clip)
            td.sess.run(td.disc_minimize, feed_dict = feed_dict)

        # Update generator
        g_iters = 5
        for _ in range(0, g_iters):
            td.sess.run(td.gene_minimize, feed_dict = feed_dict)

        if batch % 50 == 0 or batch < 100:
            ops = [td.gene_loss, td.disc_real_loss, td.disc_fake_loss, td.learning_rate]
            # TODO: face verification
            [gene_loss, disc_real_loss, \
             disc_fake_loss, learning_rate] = td.sess.run(ops, feed_dict=feed_dict)
        
            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            print('Progress[%3d%%], ETA[%4dm], Batch [%4d], learing_rate [%.10f], \
            G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f]' %
                  (int(100*elapsed/FLAGS.train_time), FLAGS.train_time - elapsed, batch, \
                   learning_rate, \
                   gene_loss, disc_real_loss, disc_fake_loss))

            # Finished?            
            current_progress = elapsed / FLAGS.train_time
            if current_progress >= 1.0:
                done = True
            # Summary
            merge = td.sess.run(summarie_op, feed_dict = feed_dict)
            td.summary_writer.add_summary(merge, batch)
            td.summary_writer.flush()
           
        if batch % FLAGS.summary_period == 0:
            # Show progress with test features
            feed_dict = {td.gene_input_pl : test_feature}
            summary_image = td.sess.run(image, feed_dict=feed_dict)
            _summarize_progress(summary_image, dirs.imgs_dir, batch)
           
        if batch % FLAGS.checkpoint_period == 0:
            # Save checkpoint
            _save_checkpoint(td, dirs.ckpt_dir, batch)
        # debug
        # _save_checkpoint(td, batch)

    _save_checkpoint(td, dirs.ckpt_dir, batch)
    print('Finished training!')

def train_model_gan(train_data):
    td = train_data
    dirs = td.dirs
    # Cache test features and labels (they are small)
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])
    num_samples = FLAGS.test_vectors
    image = _get_summary_image(test_feature, test_label, td.gene_output, num_samples)

    summarie_op = tf.summary.merge_all()

    batch = 0
    if FLAGS.resume:
        ckpt_path = td.ckpt.model_checkpoint_path
        td.saver.restore(td.sess, ckpt_path)
        batch = int(ckpt_path.split('-')[-1])
        print('Resume from ', ckpt_path)
    else:
        # tensorflow version 1.0
        init = tf.global_variables_initializer()
        td.sess.run(init)
        print('\t Training from scratch!')
        
    tf.get_default_graph().finalize()
    start_time  = time.time()
    done  = False
    while not done:
        batch += 1
        gene_loss = disc_real_loss = disc_fake_loss = -1.234
        
        train_features, gene_input, real_images = td.sess.run([td.train_features, \
                                                               td.gene_input, \
                                                               td.train_labels])
        feed_dict = {td.train_features_pl : train_features, \
                     td.gene_input_pl : gene_input, \
                     td.real_images_pl : real_images, \
                     td.global_step_pl : batch}

        # Update discriptor
        d_iters = 5
        # if batch % 500 == 0 or batch < 25:
        #     d_iters = 100
        for _ in range(0, d_iters):
            td.sess.run(td.disc_minimize, feed_dict = feed_dict)

        # Update generator
        g_iters = 1
        for _ in range(0, g_iters):
            td.sess.run(td.gene_minimize, feed_dict = feed_dict)

        if batch % 50 == 0 or batch < 100:
            ops = [td.gene_loss, td.disc_real_loss, td.disc_fake_loss, td.learning_rate]
            # TODO: face verification
            [gene_loss, disc_real_loss, \
             disc_fake_loss, learning_rate] = td.sess.run(ops, feed_dict=feed_dict)
        
            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            print('Progress[%3d%%], ETA[%4dm], Batch [%4d], learing_rate [%.10f], \
            G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f]' %
                  (int(100*elapsed/FLAGS.train_time), FLAGS.train_time - elapsed, batch, \
                   learning_rate, \
                   gene_loss, disc_real_loss, disc_fake_loss))

            # Finished?            
            current_progress = elapsed / FLAGS.train_time
            if current_progress >= 1.0:
                done = True
            # Summary
            merge = td.sess.run(summarie_op, feed_dict = feed_dict)
            td.summary_writer.add_summary(merge, batch)
            td.summary_writer.flush()
           
        if batch % FLAGS.summary_period == 0:
            # Show progress with test features
            feed_dict = {td.gene_input_pl : test_feature}
            summary_image = td.sess.run(image, feed_dict=feed_dict)
            _summarize_progress(summary_image, dirs.imgs_dir, batch)
           
        if batch % FLAGS.checkpoint_period == 0:
            # Save checkpoint
            _save_checkpoint(td, dirs.ckpt_dir, batch)
        # debug
        # _save_checkpoint(td, batch)

    _save_checkpoint(td, dirs.ckpt_dir, batch)
    print('Finished training!')

