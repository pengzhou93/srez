from inputs.prepare import prepare_res_gan_dirs, ScopeData, \
    get_nn_bi_summary_image, summarize_progress, \
    save_checkpoint
from inputs import celebA
from inputs import utils
from networks import residual_gan_model
from networks import srcnn_model
import pywt
from scipy import misc
import numpy as np
import os
import pprint
import time
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
#DATA_PATH = "/media/machlearn/cong/ILSVRC/Data/CLS-LOC/test"
DATA_PATH = "/media/machlearn/cong/ILSVRC/Data/CLS-LOC/test"
TEST_DATA_PATH = "Set14_train"
DP_TEST_PATH ='test'

def train_model():

    # Create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)

    # Prepare directories
    dirs = prepare_res_gan_dirs()

    # Setup async input queues
    # train_features : down sample images(4x)   train_labels : original images(64 64 3)
    train_list= celebA.get_train_list(DATA_PATH)
    [train_features,  train_labels] = celebA.get_batch_inputs(sess,DATA_PATH)
    [test_features,  test_labels] = celebA.get_test_input(sess,TEST_DATA_PATH,DP_TEST_PATH)

    # Add some noise during training (think denoising autoencoders)
    # noise_level = .03
    # gene_input = train_features + \
    #                        tf.random_normal(train_features.get_shape(), stddev=noise_level)

    [_, rows, cols, channels] = train_features.shape.as_list()
    [_, rows_label, cols_label, _] = train_labels.shape.as_list()

    # Placeholder for subpixel_input
    subpixel_input_pl = tf.placeholder(tf.float32,
                                       shape = [None, None, None, channels],
                                       name = 'subpixel_input_pl')
    # Placeholder for train_labels
    train_features_pl = tf.placeholder(tf.float32, \
                                       shape=[None, None, None, channels], \
                                       name = 'train_features_pl')

    real_images_pl = tf.placeholder(tf.float32, \
                                 shape = [None, None, None, channels], \
                                 name = 'real_images_pl')
    # Placeholder for gene_input
    wavelet_coeff_features_pl = tf.placeholder(tf.float32,
                                   shape=[None, None, None, 3], \
                                   name = 'wavelet_coeff_pl')
    wavelet_coeff_labels_pl = tf.placeholder(tf.float32,
                                   shape=[None, None, None, 3], \
                                   name = 'wavelet_coeff_labels_pl')
    gene_output_pl = tf.placeholder(tf.float32,
                                   shape=[None, None, None, 1], \
                                   name = 'gene_output_pl')
    # Placeholder for residual image
    # residual_images_pl = tf.placeholder(tf.float32,
    #                             shape = [FLAGS.batch_size, rows_label, cols_label, channels],
    #                             name = 'residual_images_pl')

    # Create and initialize model
    [gene_LL, gene_LL_vars,gene_wavelet, gene_wavelet_ini,gene_var_list, \
     disc_real_output, disc_fake_output, disc_var_list] = \
            residual_gan_model.create_residual_gan_model(sess, train_features_pl,
                                                         wavelet_coeff_features_pl,wavelet_coeff_labels_pl, real_images_pl)
    


    # clip
    with tf.variable_scope('d_clip') as _:
         d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in disc_var_list]

    gene_LL_loss = residual_gan_model.gene_LL_network_loss(gene_LL,real_images_pl)
    # generator loss
    gene_loss = residual_gan_model.wgan_generator_loss(disc_fake_output,gene_wavelet,gene_wavelet_ini,
                                                       wavelet_coeff_labels_pl,
                                                        real_images_pl,FLAGS.gene_l1_factor)
    # Summary
    tf.summary.scalar('loss/gene_loss', gene_loss)

    # discriminator loss
    disc_real_loss, disc_fake_loss = \
                                     residual_gan_model.wgan_discriminator_loss(
                                         disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
    # Summary
    tf.summary.scalar('loss/disc_real_loss', disc_real_loss)
    tf.summary.scalar('loss/disc_fake_loss', disc_fake_loss)
    tf.summary.scalar('loss/disc_loss', disc_loss)
    
    global_step_pl = tf.placeholder(dtype = tf.int64, name = 'global_step_pl')
    # Optimizer subpixel
    starter_subpixel_lr = FLAGS.subpixel_learning_rate_start
    gene_LL_lr = tf.train.exponential_decay(starter_subpixel_lr, global_step_pl,
                                             FLAGS.subpixel_decay_steps,
                                             FLAGS.subpixel_decay_rate,staircase = True)
    gene_LL_lr_pb = tf.summary.scalar('lr/gene_LL_lr', gene_LL_lr)
    gene_LL_minimize = residual_gan_model.gene_LL_network_optimizer(gene_LL_loss,gene_LL_vars,
                                                                      gene_LL_lr)
    
    # Optimizer GAN
    starter_gan_lr = FLAGS.gan_learning_rate_start
    gan_lr = tf.train.exponential_decay(starter_gan_lr, global_step_pl,
                                        FLAGS.gan_decay_steps,
                                        FLAGS.gan_decay_rate, staircase=True)
    tf.summary.scalar('lr/gan_lr', gan_lr)
    (gene_minimize, disc_minimize) = \
            residual_gan_model.wgan_adam_optimizers(gene_loss, gene_var_list,
                                                   disc_loss, disc_var_list,
                                                   gan_lr)

    # srcnn = srcnn_model.SRCNN(sess,
    #               image_size=None,
    #               label_size=None,
    #               batch_size=None,
    #               c_dim=3,
    #               checkpoint_dir=None,
    #               sample_dir=None)
    # srcnn.build_model(1e-4,250,240)
    #srcnn_output = srcnn.pred

    # Save model
    ckpt = tf.train.get_checkpoint_state(dirs.ckpt_dir)
    saver = tf.train.Saver(max_to_keep = 3)

    summary_writer = tf.summary.FileWriter(dirs.log_dir, tf.get_default_graph())
    
    # Train model
    train_data = ScopeData(locals())
    return train_data


def train():
    td = train_model()
    dirs = td.dirs
    # Cache test features and labels (they are small)
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])
    num_samples = FLAGS.test_vectors
    # get nearest neighbor and bicubic images for test
    #sum_images = get_nn_bi_summary_image(test_feature, test_label)

    #subpixel_merge_op = tf.summary.merge([td.subpixel_loss_pb, td.subpixel_lr_pb])
    summarie_op = tf.summary.merge_all()

    batch = 0
    if FLAGS.resume:
        ckpt_path = td.ckpt.model_checkpoint_path
        #ckpt_path = "./results/shortcut_gen/l1_0.99/checkpoint/ckpt-83600"
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

    # if subpixel_pretrain:
    #     subpixel_iters = 200000
    #     tf.logging.info('Pretrain subpixel network %d times'%subpixel_iters)
    #
    #     for step in range(subpixel_iters):
    #         train_features, real_images = td.sess.run([td.train_features, \
    #                                                    td.train_labels])
    #         subpixel_input = train_features
    #         #feed_dict = {td.subpixel_input_pl : subpixel_input, \
    #                      #td.real_images_pl : real_images, \
    #                      #td.global_step_pl : step, td.train_features_pl : train_features}
    #
    #         #[_, subpixel_loss, subpixel_lr] = td.sess.run([td.subpixel_minimize,
    #                                                        #td.subpixel_loss,
    #                                                        #td.subpixel_lr],
    #                                                       #feed_dict = feed_dict)
    #
    #         if step % FLAGS.summary_period == 0:
    #         # if True:
    #             print("step[%d/%d]]"%(step, subpixel_iters,)
    #             #feed_dict = {td.subpixel_input_pl : test_feature}
    #             #subpixel_output = td.sess.run(td.subpixel_output,
    #                                           #feed_dict = feed_dict)
    #             tmp_file = os.path.join(dirs.imgs_lf_dir, 'subpixel_%d.png'%step)
    #             utils.save_images([test_label], (16, 2), tmp_file)
    #             # summary
    #             feed_dict = {td.subpixel_input_pl : subpixel_input, \
    #                          td.real_images_pl : real_images, \
    #                          td.global_step_pl : step}
    #             merge = td.sess.run(subpixel_merge_op, feed_dict = feed_dict)
    #             #td.summary_writer.add_summary(merge, step)
    #             # snapshot
    #         if step % FLAGS.checkpoint_period == 0:
    #         # if True:
    #             save_checkpoint(td, dirs.ckpt_dir, step)
    #
    #     batch = step
        
    tf.logging.info('Begining of joint training.')
    while not done:
        batch += 1
        train_features, real_images = td.sess.run([td.train_features, \
                                                   td.train_labels])

        cA, cD = pywt.dwt2(train_features, 'haar', axes=(-3, -2))
        CA, CD = pywt.dwt2(real_images, 'haar', axes=(-3, -2))
        cD = np.concatenate(cD, axis=3)
        CD = np.concatenate(CD, axis=3)
        gene_LL_feed_dict = {td.train_features_pl : train_features,
                             td.real_images_pl:CA,
                             td.global_step_pl:batch}
                              #td.real_images_pl : real_images, \
                              #td.global_step_pl : batch}
        
        # Update subpixel network
        sub_iters = 2
        for _ in range(0, sub_iters):
            td.sess.run(td.gene_LL_minimize,feed_dict = gene_LL_feed_dict)

        #gene_LL = td.sess.run(td.gene_LL,
                            #feed_dict = gene_LL_feed_dict)
        #cA,cD = pywt.dwt2(train_features, 'haar',axes=(-3, -2))
        #CA,CD = pywt.dwt2(real_images,'haar',axes=(-3, -2))
        #cD = np.concatenate(cD,axis=3)
        #CD = np.concatenate(CD,axis=3)
        #cD = misc.imresize(cD,[16,64,64,3],'bicubic')
        #CD = misc.imresize(CD,[16,64,64,3],'bicubic')
        #subpixel_dwt = (subpixel_dwt[:,:,:,0,None],subpixel_dwt[:,:,:,1,None],subpixel_dwt[:,:,:,2,None])
        #subpiexel_dwt = tf.reshape(subpixel_dwt,[3,16,32,32,1])
        #subpixel_output = pywt.idwt2((cA,subpixel_dwt),'haar',mode='sym',axes=[-3,-2])
        # residual_images = real_images - subpixel_images
        
        gan_feed_dict = {td.train_features_pl : train_features,
                         td.real_images_pl : CA,
                         td.global_step_pl : batch,
                         td.wavelet_coeff_features_pl : cD,
                         td.wavelet_coeff_labels_pl:CD}
        # Update discriptor
        d_iters = 5
        for _ in range(0, d_iters):
            td.sess.run(td.disc_minimize, feed_dict = gan_feed_dict)
            td.sess.run(td.d_clip)
        # Update generator
        g_iters = 1
        for _ in range(0, g_iters):
            #gene_wavelet = (gene_wavelet[:, :, :, 0, None], gene_wavelet[:, :, :, 1, None], gene_wavelet[:, :, :, 2, None])
            #gene_output = pywt.idwt2((gene_LL,gene_wavelet),'haar',mode='sym',axes=[-3,-2])
            td.sess.run(td.gene_minimize, feed_dict = gan_feed_dict)

        if batch % 50 == 0 or batch < 50:
        # if True:
            sum_feed_dict = {td.real_images_pl : CA, \
                             td.global_step_pl : batch, \
                             td.train_features_pl : train_features, \
                             td.wavelet_coeff_features_pl:cD, \
                             td.wavelet_coeff_labels_pl:CD}
            ops = [td.disc_real_loss, \
                   td.disc_fake_loss, td.gan_lr]
            # TODO: face verification
            [disc_real_loss, \
             disc_fake_loss, gan_lr] = td.sess.run(ops, feed_dict=sum_feed_dict)
        
            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            print('Progress[%3d%%], ETA[%4dm], Batch [%4d]\n' 
                  '\tgan_lr[%.10f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f]' %
            (int(100*elapsed/FLAGS.train_time), FLAGS.train_time - elapsed, batch, \
             gan_lr, disc_real_loss, disc_fake_loss))

            # Finished?            
            current_progress = elapsed / FLAGS.train_time
            if current_progress >= 1.0:
                done = True
            # Summary
            merge = td.sess.run(summarie_op, feed_dict = sum_feed_dict)
            td.summary_writer.add_summary(merge, batch)
           
        if batch % FLAGS.summary_period == 0:
        # if True:
            sum_feed_dict = {td.train_features_pl : test_feature}
            #test_subpixel_output = td.sess.run(td.subpixel_output,
                                               #feed_dict = sum_feed_dict)
            
            # generated low frequency images
            #test_subpixel_images = test_subpixel_output
            test_cA,test_cD = pywt.dwt2(test_feature,'haar',axes=[-3,-2])
            test_cD = np.concatenate(test_cD, axis=3)
            test_CA, test_CD = pywt.dwt2(test_label,'haar',axes=[-3,-2])
            #test_cD = misc.imresize(test_cD,[16,64,64,3],'bicubic')
            #test_srcnn_output = td.sess.run(td.srcnn_output,{td.srcnn.images: test_cD})
            #test_srcnn_output = (test_srcnn_output[:, :, :, 0, None], test_srcnn_output[:, :, :, 1, None], test_srcnn_output[:, :, :, 2, None])
            #test_subpixel_restore = pywt.idwt2((test_cA,test_srcnn_output),'haar',mode='sym',axes=[-3,-2])
            #test_subpixel_output = np.concatenate((test_feature,test_srcnn_output),axis=3)
            sum_feed_dict.update({td.wavelet_coeff_features_pl:test_cD})
            # generated super resolution images
            test_gene_LL,test_gene_wavelet,test_gene_wavelet_ini = td.sess.run([td.gene_LL,td.gene_wavelet,td.gene_wavelet_ini],
                                           feed_dict = sum_feed_dict)
            test_gene_wavelet = (test_gene_wavelet[:, :, :, 0, None], test_gene_wavelet[:, :, :, 1, None], test_gene_wavelet[:, :, :, 2, None])
            test_gene_output = pywt.idwt2((test_gene_LL, test_gene_wavelet), 'haar', mode='sym', axes=[-3, -2])
            test_gene_output = np.clip(test_gene_output,0,1)
            #test_gene_output_images = (test_gene_output + 1) / 2
            # test_gene_output_images = np.clip(test_gene_output, 0, 1)

            # residual images
            #test_residual_images = test_gene_output_images - test_subpixel_images
            #test_residual_images = utils.inverse_transform(test_residual_images)
            #test_residual_images_clip = np.clip(test_residual_images, 0, 1)
            
            # get nearest and bicubic images for test
            #test_nn, test_bic = td.sess.run(sum_images)

            test_file = os.path.join(dirs.imgs_dir, '%d.png'%batch)
            # utils.save_images([test_nn, test_label], (16, 2), test_file)
            utils.save_images([ test_gene_output, test_label], (1, 2), test_file)
            test_file_2 = os.path.join(dirs.imgs_dir, 'wavelet_%d.png' % batch)
            utils.save_images([test_gene_wavelet[0],test_CD[0]],(1,2),test_file_2)

            pass
        
        if batch % FLAGS.checkpoint_period == 0:
            # Save checkpoint
            save_checkpoint(td, dirs.ckpt_dir, batch)

    save_checkpoint(td, dirs.ckpt_dir, batch)
    print('Finished training!')



