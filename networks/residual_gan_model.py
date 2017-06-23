from .srez_model import Model, _discriminator_model
import numpy as np
import tensorflow as tf
from layers import ops
from layers import subpixel

slim = tf.contrib.slim

def create_residual_gan_model(sess, subpixel_input_pl,
                              gene_input_pl, real_images_pl,
                              residual_images_pl, channels = 3):
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
    # generate low frequency image
    # subpixel_output, subpixel_var_list = subpixel_network(subpixel_input_pl)    
    # subpixel_output, subpixel_var_list = subpixel_network_v2(subpixel_input_pl)
    subpixel_output, subpixel_var_list = subpixel_network_v3(subpixel_input_pl)

    # generate high frequency image
    gene_output, gene_var_list = generator_network(gene_input_pl, channels)

    # Discriminator with real data
    disc_real_input = tf.identity(residual_images_pl, name='disc_real_input')
    # TBD: Is there a better way to instance the discriminator?
    with tf.variable_scope('disc') as scope:
        disc_real_output, disc_var_list = \
                _discriminator_model(sess, gene_input_pl, disc_real_input)

        scope.reuse_variables()
            
        disc_fake_output, _ = _discriminator_model(sess, gene_input_pl, gene_output)

    return [subpixel_output, subpixel_var_list,
            gene_output, gene_var_list,
            disc_real_output, disc_fake_output, disc_var_list]

def generator_network(features, channels):
    # Upside-down all-convolutional resnet

    mapsize = 3
    res_units  = [256, 128, 96]

    old_vars = tf.global_variables()

    # See Arxiv 1603.05027
    model = Model('GEN_subpixel', features)
    with tf.variable_scope('GEN_subpixel'):
        for ru in range(len(res_units)-1):
            nunits  = res_units[ru]
    
            for j in range(2):
                model.add_residual_block(nunits, mapsize=mapsize)
    
            # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
            # and transposed convolution
            # model.add_upscale()
            
            model.add_batch_norm()
            model.add_relu()
            model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)
    
        # Finalization a la "all convolutional net"
        nunits = res_units[-1]
        model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()
    
        model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()
    
        # Last layer is sigmoid with no batch normalization
        # upscaling ratio
        # ratio = 4               
        # model.add_conv2d(channels * ratio * ratio, mapsize=1, stride=1, stddev_factor=1.)
        # Spatial upscale : subpixel
        # model.add_upscale_subpixel(r = ratio, color=True)

        model.add_conv2d(channels, mapsize = 3, stride = 1, stddev_factor = 1.)
        # model.add_sigmoid()

        # residual images
        model.add_tanh()

        # Add shortcut layer
        output = tf.add(features, model.get_output(), name = "gene_output")
        model.outputs.append(output)
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), gene_vars

def gan_generator_loss(gene_output, residual_images_pl,
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
        gene_l1_loss  = tf.reduce_mean(tf.abs(gene_output - residual_images_pl),
                                       name='gene_l1_loss')
    
        gene_loss     = tf.add((1.0 - gene_l1_factor) * gene_ce_loss,
                               gene_l1_factor * gene_l1_loss, name='gene_loss')
    
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
        net = slim.conv2d(net, 3*16, kernel_size = (3, 3), stride = 1,
                          padding='SAME', activation_fn = None,
                          scope = 'conv3')        
        net = subpixel.PS(net, r = 4, color = True)
    new_vars  = tf.global_variables()
    subpixel_var_list = list(set(new_vars) - set(old_vars))

    return tf.tanh(net), subpixel_var_list      # [-1, 1]

def subpixel_network_loss(subpixel_output, real_images_pl):
    with tf.variable_scope("subpixel_network_loss"):
        real_images_pl = (real_images_pl - 0.5) * 2
        loss = tf.reduce_mean(tf.abs(subpixel_output - real_images_pl))
        # loss = tf.reduce_mean(tf.square(subpixel_output - real_images_pl))
    return loss

def subpixel_network_optimizer(loss, var_list, lr, beta1 = 0.5):
    subpixel_minimize = tf.train.AdamOptimizer(lr, beta1).minimize(loss, var_list = var_list,
                                                                   name = 'subpixel_minimize')
    return subpixel_minimize
