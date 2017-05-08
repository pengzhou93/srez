from inputs.prepare import prepare_dirs, ScopeData, get_summary_image, summarize_progress, \
    save_checkpoint
from inputs import celebA
from networks import subpixel_model

import numpy as np
import os
import pprint
import time
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def gan_subpixel_train():
    print('\t Train subpixel gan')

    # Create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)

    # Prepare directories
    dirs = prepare_dirs()
    summary_writer = tf.summary.FileWriter(dirs.log_dir, sess.graph)

    # Setup async input queues
    # train_features : down sample images(4x)   train_labels : original images(64 64 3)
    [train_features, train_labels, \
    test_features,  test_labels]  = celebA.get_batch_inputs(sess, FLAGS.dataset)

    # Add some noise during training (think denoising autoencoders)
    noise_level = .03
    gene_input = train_features + \
                           tf.random_normal(train_features.get_shape(), stddev=noise_level)

    [_, rows, cols, channels] = train_features.shape.as_list()
    [_, rows_label, cols_label, _] = train_labels.shape.as_list()
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
            subpixel_model.create_subpixel_model(sess, gene_input_pl, real_images_pl)
    
    # generator loss
    gene_loss = subpixel_model.gan_generator_subpixel_loss(gene_output, real_images_pl,
                                                           disc_fake_output,
                                                           FLAGS.gene_l1_factor)

    # Summary
    tf.summary.scalar('gene_loss', gene_loss)
    summary_gene_loss = tf.summary.FileWriter(dirs.log_dir + '/gene_loss')
    summary_gene_loss.add_graph(gene_loss.graph)

    # discriminator loss
    disc_real_loss, disc_fake_loss = \
                                     subpixel_model.gan_discriminator_loss(
                                         disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
    # Summary
    tf.summary.scalar('disc_real_loss', disc_real_loss)
    tf.summary.scalar('disc_fake_loss', disc_fake_loss)
    tf.summary.scalar('disc_loss', disc_loss)
    summary_disc_loss = tf.summary.FileWriter(dirs.log_dir + '/disc_loss')
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
            subpixel_model.gan_adam_optimizers(gene_loss, gene_var_list,
                                               disc_loss, disc_var_list,
                                               learning_rate, FLAGS.learning_beta1)

    # Save model
    ckpt = tf.train.get_checkpoint_state(dirs.ckpt_dir)
    saver = tf.train.Saver(max_to_keep = 2)

    # Train model
    train_data = ScopeData(locals())
    _gan_train(train_data)

def _gan_train(train_data):
    td = train_data
    dirs = td.dirs
    # Cache test features and labels (they are small)
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])
    num_samples = FLAGS.test_vectors
    images = get_summary_image(test_feature, test_label, td.gene_output, num_samples)

    summarie_op = tf.summary.merge_all()

    batch = 0
    if FLAGS.resume:
        ckpt_path = td.ckpt.model_checkpoint_path
        td.saver.restore(td.sess, ckpt_path)
        batch = int(ckpt_path.split('-')[-1])
        print('\t Resume from ', ckpt_path)
    else:
        init = tf.global_variables_initializer()
        td.sess.run(init)
        print('\t Training from scratch!')
    # Graph definition finish
    tf.get_default_graph().finalize()
    start_time  = time.time()
    done  = False
    while not done:
        batch += 1
        train_features, gene_input, real_images = td.sess.run([td.train_features, \
                                                               td.gene_input, \
                                                               td.train_labels])
        feed_dict = {td.train_features_pl : train_features, \
                     td.gene_input_pl : gene_input, \
                     td.real_images_pl : real_images, \
                     td.global_step_pl : batch}

        # Update discriptor
        d_iters = 5
        for _ in range(0, d_iters):
            td.sess.run(td.disc_minimize, feed_dict = feed_dict)

        # Update generator
        g_iters = 1
        for _ in range(0, g_iters):
            td.sess.run(td.gene_minimize, feed_dict = feed_dict)

        if batch % 50 == 0 or batch < 50:
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
            summary_image = td.sess.run(images, feed_dict=feed_dict)
            summarize_progress(summary_image, dirs.imgs_dir, batch)
           
        if batch % FLAGS.checkpoint_period == 0:
            # Save checkpoint
            save_checkpoint(td, dirs.ckpt_dir, batch)

    save_checkpoint(td, dirs.ckpt_dir, batch)
    print('Finished training!')

