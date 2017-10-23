from .srez_model import Model, _discriminator_model
import numpy as np
import tensorflow as tf
from layers import ops
from layers import subpixel
import pywt
FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

def create_residual_gan_model(sess, train_features_pl,
                            wavelet_coeff_pl, wavelet_coeff_labels_pl,real_images_pl,
                              channels = 1):
    """
     Create gene and disc networks
    
    Parameters:
    ----------
     gene_input_pl : tf.placeholder
         Input of generator, shape : (16, 16, 16, 3)
     real_images_pl : tf.placeholder
         Input of discriminator, shape : (16, 64, 64, 3)
    Returns:
    ----------
     list 
    """

    gene_LL, gene_LL_vars = gene_LL_network(train_features_pl,channels)
    # subpixel_output, subpixel_var_list = subpixel_network_v2(subpixel_input_pl)
    # generate low frequency image [0, 1]

    # generate high frequency image [0, 1]
    gene_wavelet, gene_wavelet_ini,gene_var_list = generator_network(gene_LL,wavelet_coeff_pl,channels)

    # Discriminator with real data
    disc_real_input = tf.identity(wavelet_coeff_labels_pl, name='disc_real_input')
    # TBD: Is there a better way to instance the discriminator?
    with tf.variable_scope('disc') as scope:
        # limit the range of input of disc in [-1, 1]
        #disc_real_input = 2*disc_real_input - 1
        disc_real_output, disc_var_list = \
                _discriminator_model(sess, disc_real_input)

        scope.reuse_variables()
            
        disc_fake_output, _  = _discriminator_model(sess, gene_wavelet)

    return [gene_LL, gene_LL_vars,gene_wavelet,gene_wavelet_ini,gene_var_list,
            disc_real_output, disc_fake_output, disc_var_list]

#def feature_extractor(image,wavecoeffs):

def add_conv2d(input, prev_units,num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a 2D convolutional layer."""

        assert len(input.get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        with tf.variable_scope('conv2d'):
            #prev_units = input.shape[3]

            # Weight term and convolution
            kernel = tf.Variable(tf.truncated_normal([mapsize, mapsize, prev_units, num_units], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')

            # initw = input._glorot_initializer_conv2d(prev_units, num_units,
            #                                         mapsize,
            #                                         stddev_factor=stddev_factor)
            # weight = tf.get_variable('weight', initializer=initw)
            out = tf.nn.conv2d(input, kernel,
                               strides=[1, stride, stride, 1],
                               padding='SAME')

            # Bias term
            initb = tf.constant(0.0, shape=[num_units])
            bias = tf.Variable(initb,name="bias")
            out = tf.nn.bias_add(out, bias)

        return out

def gene_LL_network(features,channels):
    old_vars = tf.global_variables()
    assert len(features.get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"
    with tf.variable_scope('subpixel'):
        net_1 = add_conv2d(features, 1,64,mapsize=3)
        net_1= tf.tanh(net_1)
        net_1 = add_conv2d(net_1, 64,64,mapsize=3)
        net_1 = tf.tanh(net_1)
        net_1 = add_conv2d(net_1, 64,4,mapsize=3)
        net_1 = tf.depth_to_space(net_1, 2, name='subpixel')
        new_vars = tf.global_variables()
        gene_LL_vars = list(set(new_vars) - set(old_vars))
    return net_1,gene_LL_vars

def generator_network(gene_LL, wavelet_coeff,channels):
    # Upside-down all-convolutional resnet


    mapsize = 3
    #res_units  = [256, 128, 96]

    old_vars = tf.global_variables()
    #assert len(features.get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"
    # See Arxiv 1603.05027
    with tf.variable_scope('GEN'):
        #net_1 = add_conv2d(features, 1,64,mapsize=3)
        #net_1= tf.tanh(net_1)
        #net_1 = add_conv2d(net_1, 64,64,mapsize=3)
        #net_1 = tf.tanh(net_1)
        #net_1 = add_conv2d(net_1, 64,4,mapsize=3)
        #net_1 = tf.depth_to_space(net_1, 2, name='subpixel')
        #net_1 = tf.sigmoid(net_1)
        with tf.variable_scope('subpixel_wavelet'):
            net_2 = add_conv2d(wavelet_coeff, 3,64,mapsize=3)
            net_2 = tf.nn.tanh(net_2)
            net_2 = add_conv2d(net_2, 64,64,mapsize=3)
            net_2 = tf.tanh(net_2)
            net_2 = add_conv2d(net_2, 64,48,mapsize=3)
            net_2 = tf.depth_to_space(net_2, 4, name='subpixel_2')
        wavelet_ini = net_2
        with tf.variable_scope('wavelet_residual'):
            with tf.variable_scope('convd_tanh'):
                net = tf.concat([gene_LL,net_2],axis=3)
                net = add_conv2d(net, 4,64,mapsize=9)
                net = tf.tanh(net)
                bypass = net
            for ru in range(9):
                with tf.variable_scope('residual_unit_%d'%ru):
                    nunits  = 64
                    if nunits != int(net.shape[3]):
                        net = add_conv2d(net,int(net.shape[3]), nunits,mapsize=3)
                        net = tf.tanh(net)
                # Residual block
                    for i in range(2):
                        net = add_conv2d(net, nunits,nunits,mapsize=3)
                        net = tf.contrib.layers.batch_norm(net, scale=False,reuse=None, scope = 'bn_%d_%d'%(ru, i))
                        net = tf.tanh(net)

                net = tf.add(net,bypass,name='%d'%ru)

            # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
            # and transposed convolution
            # model.add_upscale()
            # Finalization a la "all convolutional net"

        net = add_conv2d(net, int(net.get_shape()[3]),3,mapsize=3)
        net = tf.contrib.layers.batch_norm(net, scale=False)
        # Worse: model.add_batch_norm()
        #net = tf.tanh(net)

        # Worse: model.add_batch_norm()
    
        # Last layer is sigmoid with no batch normalization
        # upscaling ratio
        # ratio = 4               
        # model.add_conv2d(channels * ratio * ratio, mapsize=1, stride=1, stddev_factor=1.)
        # Spatial upscale : subpixel
        # model.add_upscale_subpixel(r = ratio, color=True)

        # residual images
        # model.add_sigmoid()
        # Add shortcut layer
        output = tf.add(net, wavelet_ini)
        output = tf.tanh(output)
        #output = add_conv2d(output, 64,256,mapsize=3)
        #output = tf.depth_to_space(output, 2, name='subpixel_3')
        #output = tf.nn.relu(output)
        # output = tf.nn.sigmoid(output)

        # output_clip = tf.clip_by_value(output, 0, 1, name = "gene_output")
        # model.outputs.append(output_clip)
        # model.add_tanh()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))

    return output,net_2, gene_vars

def gan_generator_loss(gene_output, real_images_pl,
                       disc_output, gene_l1_factor):
    with tf.variable_scope("gan_generator_loss"):
        # I.e. did we fool the discriminator?
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_output,
                                                                labels = tf.ones_like(disc_output))
        gene_ce_loss  = tf.reduce_mean(cross_entropy, name='gene_ce_loss')
    
        # I.e. does the result look like the feature?
        # K = int(gene_output.get_shape()[1])//int(train_features_pl.get_shape()[1])
        # assert K == 2 or K == 4 or K == 8    
        # downscaled = _downscale(gene_output, K)
        # gene_l1_loss  = tf.reduce_mean(tf.abs(downscaled - train_features_pl), name='gene_l1_loss')    
        # gene_l2_loss  = tf.reduce_mean(tf.square(gene_output - real_images_pl), name='gene_l2_loss')
        gene_l1_loss  = tf.reduce_mean(tf.abs(gene_output - real_images_pl),
                                       name='gene_l1_loss')
    
        gene_loss     = tf.add((1.0 - FLAGS.gene_l1_factor) * gene_ce_loss,
                               FLAGS.gene_l1_factor * gene_l1_loss, name='gene_loss')
    
    return


def _downscale(images, K):
    """Differentiable image downscaling by a factor of K"""
    arr = np.zeros([K, K, 3, 1])
    arr[:, :, 0, 0] = 1.0 / (K * K)
    arr[:, :, 1, 1] = 1.0 / (K * K)
    arr[:, :, 2, 2] = 1.0 / (K * K)
    dowscale_weight = tf.constant(arr, dtype=tf.float32)

    downscaled = tf.nn.conv2d(images, dowscale_weight,
                              strides=[1, K, K, 1],
                              padding='SAME')
    return downscaled


def wgan_generator_loss(disc_output, gene_wavelet,gene_wavelet_ini,
                        wavelet_labels, real_images_pl,gene_l1_factor):
    # I.e. did we fool the discriminator?
    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_output,
    #                                                         labels = tf.ones_like(disc_output))
    # gene_ce_loss  = tf.reduce_mean(cross_entropy, name='gene_ce_loss')
    # Generator loss for wgan
    gene_wgan_loss = - tf.reduce_mean(disc_output)
    # I.e. does the result look like the feature?
    #K = int(gene_output.get_shape()[1]) // int(train_features_pl.get_shape()[1])
    #assert K == 2 or K == 4 or K == 8
    #downscaled = _downscale(gene_output, K)

    # subtract real image
    #real_images_pl = 2*real_images_pl - 1
    #gene_output = pywt.idwt2(gene_LL,gene_wavelet)
    #l1_loss = tf.reduce_mean(tf.abs(gene_LL  - real_images_pl), name='gene_l1_loss')
    #error = downscaled - train_features_pl
    #xmean = tf.reduce_mean(error, 0)
    #xnorm2 = tf.square(tf.norm(error - xmean, 'euclidean', [-3, -2]))
    #lossnorm2 = tf.reduce_mean(xnorm2)
    # gene_l1_loss=tf.reduce_mean(tf.abs(xmean))+lossnorm2/(16*16)
    #gene_l1_loss = l1_loss + lossnorm2 / (16 * 16)
    #gene_l1_loss = l1_loss
    wavelet_loss = tf.reduce_mean(tf.square(gene_wavelet-wavelet_labels))
    #feature_loss = tf.reduce_mean(tf.norm(real_feature_loss-fake_feature_loss,'euclidean', [-3,-2]))

    gene_loss = tf.add((1.0 - gene_l1_factor) * gene_wgan_loss,wavelet_loss,
                        name='gene_loss')
    return gene_loss

def gan_discriminator_loss(disc_real_output, disc_fake_output):
    with tf.variable_scope("gan_disc_loss"):
        # I.e. did we correctly identify the input as real or not?
        cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits( \
                                                logits = disc_real_output,
                                                labels = tf.ones_like(disc_real_output))
        disc_real_loss     = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')
        
        cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits( \
                                                logits = disc_fake_output,
                                                labels = tf.zeros_like(disc_fake_output))
        disc_fake_loss     = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')
    
        # Discriminator loss for wgan
        # disc_real_loss = - tf.reduce_mean(disc_real_output)
        # disc_fake_loss = tf.reduce_mean(disc_fake_output)
    return disc_real_loss, disc_fake_loss


def wgan_discriminator_loss(disc_real_output, disc_fake_output):
    # I.e. did we correctly identify the input as real or not?
    # cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_real_output,
    #                                                              labels = tf.ones_like(disc_real_output))
    # disc_real_loss     = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')

    # cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake_output,
    #                                                              labels = tf.zeros_like(disc_fake_output))
    # disc_fake_loss     = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')
    # Discriminator loss for wgan
    disc_real_loss = - tf.reduce_mean(disc_real_output)
    disc_fake_loss = tf.reduce_mean(disc_fake_output)
    return disc_real_loss, disc_fake_loss

def gan_adam_optimizers(gene_loss, gene_var_list,
                        disc_loss, disc_var_list,
                        learning_rate, learning_beta1):    
    # TBD: Does this global step variable need to be manually incremented? I think so.
    # global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
     
    gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=learning_beta1,
                                       name='gene_optimizer')
    disc_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=learning_beta1,
                                       name='disc_optimizer')

    gene_minimize = gene_opti.minimize(gene_loss, var_list = gene_var_list, \
                                       name = 'gene_loss_minimize')
    
    disc_minimize = disc_opti.minimize(disc_loss, var_list=disc_var_list, \
                                           name = 'disc_loss_minimize')#, global_step = global_step)
    
    return (gene_minimize, disc_minimize)


def wgan_adam_optimizers(gene_loss, gene_var_list,
                           disc_loss, disc_var_list,
                           learning_rate):
    # TBD: Does this global step variable need to be manually incremented? I think so.
    # global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')

    # gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate_pl,
    #                                    beta1=FLAGS.learning_beta1,
    #                                    name='gene_optimizer')
    # disc_opti = tf.train.AdamOptimizer(learning_rate=learning_rate_pl,
    #                                    beta1=FLAGS.learning_beta1,
    #                                    name='disc_optimizer')
    gene_opti = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          name='gene_optimizer_RMS')
    disc_opti = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          name='disc_optimizer_RMS')

    gene_minimize = gene_opti.minimize(gene_loss, var_list=gene_var_list, \
            name = 'gene_loss_minimize')

    disc_minimize = disc_opti.minimize(disc_loss, var_list=disc_var_list, \
                                       name='disc_loss_minimize')  # , global_step = global_step)

    return (gene_minimize, disc_minimize)


def subpixel_network(x):
    """Create subpixel network
    """
    old_vars = tf.global_variables()
    with tf.variable_scope('subpixel_network'):
        net = slim.conv2d_transpose(x, 64, kernel_size = (1, 1), stride = 1,
                                    padding='SAME', activation_fn = None,
                                    scope = 'deconv1')
        net = ops.lrelu(net)
        net = slim.conv2d_transpose(net, 64, kernel_size = (5, 5), stride = 1,
                                    padding='SAME', activation_fn = None,
                                    scope = 'deconv2')        
        net = ops.lrelu(net)
        net = slim.conv2d_transpose(net, 3*16, kernel_size = (5, 5), stride = 1,
                                    padding='SAME', activation_fn = None,
                                    scope = 'deconv3')        
        net = subpixel.PS(net, r = 4, color = True)
    new_vars  = tf.global_variables()
    subpixel_var_list = list(set(new_vars) - set(old_vars))

    return tf.tanh(net), subpixel_var_list      # [-1, 1]

def subpixel_network_v2(x):
    old_vars = tf.global_variables()
    h0, h0_w, h0_b = ops.deconv2d(x,
                                  [16, 16, 16, 64],
                                  k_h=1, k_w=1, d_h=1, d_w=1,
                                  name='g_h0',
                                  with_w=True)
    h0 = ops.lrelu(h0)

    h1, h1_w, h1_b = ops.deconv2d(h0,
                                  [16, 16, 16, 64],
                                  name='g_h1',
                                  d_h=1, d_w=1,
                                  with_w=True)
    h1 = ops.lrelu(h1)

    h2, h2_w, h2_b = ops.deconv2d(h1,
                                  [16, 16, 16, 3*16],
                                  d_h=1, d_w=1,
                                  name='g_h2',
                                  with_w=True)
    h2 = subpixel.PS(h2, r = 4, color=True)

    new_vars  = tf.global_variables()
    subpixel_var_list = list(set(new_vars) - set(old_vars))

    return tf.tanh(h2), subpixel_var_list      # [-1, 1]
    
def subpixel_network_v3(x):
    """Create subpixel network
    """
    old_vars = tf.global_variables()
    with tf.variable_scope('subpixel_network'):
        net = slim.conv2d(x, 64, kernel_size = (3, 3), stride = 1,
                          padding='SAME', activation_fn = None,
                          scope = 'conv1')
        net = ops.lrelu(net)
        net = slim.conv2d(net, 64, kernel_size = (3, 3), stride = 1,
                          padding='SAME', activation_fn = None,
                          scope = 'conv2')        
        net = ops.lrelu(net)
        net = slim.conv2d(net, 16, kernel_size = (3, 3), stride = 1,
                          padding='SAME', activation_fn = None,
                          scope = 'conv3')        
        net = tf.depth_to_space(net, 4, name='subpixel')
    new_vars  = tf.global_variables()
    subpixel_var_list = list(set(new_vars) - set(old_vars))

    return tf.sigmoid(net), subpixel_var_list
    # return tf.tanh(net), subpixel_var_list      # [-1, 1]

def gene_LL_network_loss(gene_LL, real_images_pl):
    with tf.variable_scope("gene_LL_network_loss"):
        loss = tf.reduce_mean(tf.square(gene_LL - real_images_pl))
        # loss = tf.reduce_mean(tf.square(subpixel_output - real_images_pl))
    return loss

def gene_LL_network_optimizer(loss, gene_LL_vars, lr, beta1 = 0.5):
    gene_LL_minimize = tf.train.AdamOptimizer(lr, beta1).minimize(loss, var_list = gene_LL_vars,
                                                                   name = 'gene_LL_minimize')
    return gene_LL_minimize
