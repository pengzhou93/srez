from inputs.prepare import prepare_res_gan_dirs, ScopeData, \
    get_nn_bi_summary_image, summarize_progress, \
    save_checkpoint
from inputs import celebA
from inputs import utils
from networks import residual_gan_model

import numpy as np
import os
import pprint
import time
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def train():

    # Create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)

    # Prepare directories
    dirs = prepare_res_gan_dirs()

    # Setup async input queues
    # train_features : down sample images(4x)   train_labels : original images(64 64 3)
    [train_features, train_labels, \
    test_features,  test_labels]  = celebA.get_batch_inputs(sess, FLAGS.dataset)

    # Add some noise during training (think denoising autoencoders)
    # noise_level = .03
    # gene_input = train_features + \
    #                        tf.random_normal(train_features.get_shape(), stddev=noise_level)

    [_, rows, cols, channels] = train_features.shape.as_list()
    [_, rows_label, cols_label, _] = train_labels.shape.as_list()

    # Placeholder for subpixel_input
    subpixel_input_pl = tf.placeholder(tf.float32,
                                       shape = [FLAGS.batch_size, rows, cols, channels],
                                       name = 'subpixel_input_pl')
    # Placeholder for train_labels
    real_images_pl = tf.placeholder(tf.float32, \
                                 shape = [FLAGS.batch_size, rows_label, cols_label, channels], \
                                 name = 'real_images_pl')
    # Placeholder for gene_input
    gene_input_pl = tf.placeholder(tf.float32,
                                   shape=[FLAGS.batch_size, rows_label, cols_label, channels], \
                                   name = 'gene_input_pl')
    # Placeholder for residual image
    # residual_images_pl = tf.placeholder(tf.float32,
    #                             shape = [FLAGS.batch_size, rows_label, cols_label, channels],
    #                             name = 'residual_images_pl')

    # Create and initialize model
    [subpixel_output, subpixel_var_list,
     gene_output, gene_var_list, \
     disc_real_output, disc_fake_output, disc_var_list] = \
            residual_gan_model.create_residual_gan_model(sess, subpixel_input_pl,
                                                         gene_input_pl, real_images_pl)
    
    # subpixel network loss
    subpixel_loss = residual_gan_model.subpixel_network_loss(subpixel_output, real_images_pl)
    subpixel_loss_pb = tf.summary.scalar('loss/subpixel_loss', subpixel_loss)
    # generator loss
    gene_loss = residual_gan_model.gan_generator_loss(gene_output, real_images_pl,
                                                      disc_fake_output,
                                                      FLAGS.gene_l1_factor)
    # Summary
    tf.summary.scalar('loss/gene_loss', gene_loss)

    # discriminator loss
    disc_real_loss, disc_fake_loss = \
                                     residual_gan_model.gan_discriminator_loss(
                                         disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
    # Summary
    tf.summary.scalar('loss/disc_real_loss', disc_real_loss)
    tf.summary.scalar('loss/disc_fake_loss', disc_fake_loss)
    tf.summary.scalar('loss/disc_loss', disc_loss)
    
    global_step_pl = tf.placeholder(dtype = tf.int64, name = 'global_step_pl')
    # Optimizer subpixel
    starter_subpixel_lr = FLAGS.subpixel_learning_rate_start
    subpixel_lr = tf.train.exponential_decay(starter_subpixel_lr, global_step_pl,
                                             FLAGS.subpixel_decay_steps,
                                             FLAGS.subpixel_decay_rate,staircase = True)
    subpixel_lr_pb = tf.summary.scalar('lr/subpixel_lr', subpixel_lr)
    subpixel_minimize = residual_gan_model.subpixel_network_optimizer(subpixel_loss,
                                                                      subpixel_var_list,
                                                                      subpixel_lr)
    
    # Optimizer GAN
    starter_gan_lr = FLAGS.gan_learning_rate_start
    gan_lr = tf.train.exponential_decay(starter_gan_lr, global_step_pl,
                                        FLAGS.gan_decay_steps,
                                        FLAGS.gan_decay_rate, staircase=True)
    tf.summary.scalar('lr/gan_lr', gan_lr)
    (gene_minimize, disc_minimize) = \
            residual_gan_model.gan_adam_optimizers(gene_loss, gene_var_list,
                                                   disc_loss, disc_var_list,
                                                   gan_lr, FLAGS.learning_beta1)

    # Save model
    ckpt = tf.train.get_checkpoint_state(dirs.ckpt_dir)
    saver = tf.train.Saver(max_to_keep = 3)

    summary_writer = tf.summary.FileWriter(dirs.log_dir, sess.graph)
    
    # Train model
    train_data = ScopeData(locals())
    _gan_train(train_data)

def _gan_train(train_data):
    td = train_data
    dirs = td.dirs
    # Cache test features and labels (they are small)
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])
    num_samples = FLAGS.test_vectors
    # get nearest neighbor and bicubic images for test
    sum_images = get_nn_bi_summary_image(test_feature, test_label)

    subpixel_merge_op = tf.summary.merge([td.subpixel_loss_pb, td.subpixel_lr_pb])
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
    subpixel_pretrain = False
    done  = False

    if subpixel_pretrain:
        subpixel_iters = 200000
        tf.logging.info('Pretrain subpixel network %d times'%subpixel_iters)
        
        for step in range(subpixel_iters):
            train_features, real_images = td.sess.run([td.train_features, \
                                                       td.train_labels])
            subpixel_input = train_features
            feed_dict = {td.subpixel_input_pl : subpixel_input, \
                         td.real_images_pl : real_images, \
                         td.global_step_pl : step}
            
            [_, subpixel_loss, subpixel_lr] = td.sess.run([td.subpixel_minimize,
                                                           td.subpixel_loss,
                                                           td.subpixel_lr],
                                                          feed_dict = feed_dict)

            if step % FLAGS.summary_period == 0:
            # if True:
                print("step[%d/%d], subpixel_loss[%f], lr[%f]"%(step, subpixel_iters,
                                                                subpixel_loss,
                                                                subpixel_lr))
                feed_dict = {td.subpixel_input_pl : test_feature}
                subpixel_output = td.sess.run(td.subpixel_output,
                                              feed_dict = feed_dict)
                # subpixel_output = utils.inverse_transform(subpixel_output)
                tmp_file = os.path.join(dirs.imgs_lf_dir, 'subpixel_%d.png'%step)
                utils.save_images([subpixel_output, test_label], (16, 2), tmp_file)
                # summary
                feed_dict = {td.subpixel_input_pl : subpixel_input, \
                             td.real_images_pl : real_images, \
                             td.global_step_pl : step}
                merge = td.sess.run(subpixel_merge_op, feed_dict = feed_dict)
                td.summary_writer.add_summary(merge, step)
                # snapshot
            if step % FLAGS.checkpoint_period == 0:
            # if True:
                save_checkpoint(td, dirs.ckpt_dir, step)
                     
        batch = step
        
    tf.logging.info('Begining of joint training.')
    while not done:
        batch += 1
        subpixel_input, real_images = td.sess.run([td.train_features, \
                                                   td.train_labels])

        subpixel_feed_dict = {td.subpixel_input_pl : subpixel_input, \
                              td.real_images_pl : real_images, \
                              td.global_step_pl : batch}
        
        # Update subpixel network
        [_, subpixel_output] = td.sess.run([td.subpixel_minimize,
                                            td.subpixel_output],
                                           feed_dict = subpixel_feed_dict)
        subpixel_images = subpixel_output
        # residual_images = real_images - subpixel_images
        
        gan_feed_dict = {td.gene_input_pl : subpixel_output,
                         td.real_images_pl : real_images, 
                         td.global_step_pl : batch}
        # Update discriptor
        d_iters = 5
        for _ in range(0, d_iters):
            td.sess.run(td.disc_minimize, feed_dict = gan_feed_dict)

        # Update generator
        g_iters = 1
        for _ in range(0, g_iters):
            td.sess.run(td.gene_minimize, feed_dict = gan_feed_dict)

        if batch % 50 == 0 or batch < 50:
        # if True:
            sum_feed_dict = {td.subpixel_input_pl : subpixel_input,
                             td.real_images_pl : real_images,
                             td.gene_input_pl : subpixel_output,
                             td.global_step_pl : batch}
            ops = [td.subpixel_loss, td.subpixel_lr,
                   td.gene_loss, td.disc_real_loss,
                   td.disc_fake_loss, td.gan_lr]
            # TODO: face verification
            [subpixel_loss, subpixel_lr,
             gene_loss, disc_real_loss, \
             disc_fake_loss, gan_lr] = td.sess.run(ops, feed_dict=sum_feed_dict)
        
            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            print('Progress[%3d%%], ETA[%4dm], Batch [%4d]\n'
                  '\tsubpixel_lr[%.10f], subpixel_loss[%3.6f]\n' 
                  '\tgan_lr[%.10f], G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f]' %
            (int(100*elapsed/FLAGS.train_time), FLAGS.train_time - elapsed, batch, \
             subpixel_lr, subpixel_loss, \
             gan_lr, gene_loss, disc_real_loss, disc_fake_loss))

            # Finished?            
            current_progress = elapsed / FLAGS.train_time
            if current_progress >= 1.0:
                done = True
            # Summary
            merge = td.sess.run(summarie_op, feed_dict = sum_feed_dict)
            td.summary_writer.add_summary(merge, batch)
           
        if batch % FLAGS.summary_period == 0:
        # if True:
            sum_feed_dict = {td.subpixel_input_pl : test_feature}
            test_subpixel_output = td.sess.run(td.subpixel_output,
                                               feed_dict = sum_feed_dict)
            
            # generated low frequency images
            test_subpixel_images = test_subpixel_output
            sum_feed_dict.update({td.gene_input_pl : test_subpixel_output})
            # generated super resolution images
            test_gene_output = td.sess.run(td.gene_output,
                                           feed_dict = sum_feed_dict)
            test_gene_output_images = np.clip(test_gene_output, 0, 1)

            # residual images
            test_residual_images = test_gene_output_images - test_subpixel_images
            test_residual_images = utils.inverse_transform(test_residual_images)
            test_residual_images_clip = np.clip(test_residual_images, 0, 1)
            
            # get nearest and bicubic images for test
            test_nn, test_bic = td.sess.run(sum_images)

            test_file = os.path.join(dirs.imgs_dir, 'subpixel_%d.png'%batch)        
            # utils.save_images([test_nn, test_label], (16, 2), test_file)
            utils.save_images([test_nn, test_bic,
                               test_subpixel_images, test_residual_images_clip,
                               test_gene_output_images, test_label], (16, 6), test_file)

            pass
        
        if batch % FLAGS.checkpoint_period == 0:
            # Save checkpoint
            save_checkpoint(td, dirs.ckpt_dir, batch)

    save_checkpoint(td, dirs.ckpt_dir, batch)
    print('Finished training!')

