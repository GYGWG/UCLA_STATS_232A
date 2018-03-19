from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf

import numpy as np
import scipy.io as sio
from six.moves import xrange
from skimage import io

from ops import *
from utils import *
import random

class VAE(object):
    def __init__(self, sess, image_size=28,
                 batch_size=100, sample_size=100, output_size=28,
                 z_dim=5, c_dim=1, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            image_size: The size of input image.
            batch_size: The size of batch. Should be specified before training.
            sample_size: (optional) The size of sampling. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [28]
            z_dim: (optional) Dimension of latent vectors. [5]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [1]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.hidden_encoder_dim = 400
        self.hidden_decoder_dim = 400
        self.z_dim = z_dim
        self.c_dim = c_dim

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def weight_variable(self, shape):
        # return shape
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial)

    def encoder(self, image, reuse=False, train=True):
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            #######################################################
            # TODO: Define encoder network structure here. op.py
            # includes some basic layer functions for you to use.
            # Please use batch normalization layer after conv layer.
            # And use 'train' argument to indicate the mode of bn.
            # The output of encoder network should have two parts:
            # A mean vector and a log(std) vector. Both of them have
            # the same dimension with latent vector z.
            #######################################################
            # MLP
            W_encoder = self.weight_variable([self.image_size**2, self.hidden_encoder_dim])
            b_encoder = self.bias_variable([self.hidden_encoder_dim])
            h_encoder = tf.nn.relu(tf.matmul(image, W_encoder) + b_encoder)
            # Convolutional Neural Network
            # x = tf.placeholder("float", shape=[None, self.image_size, self.image_size, self.c_dim])
            # conv = conv2d(x, self.image_size)
            # pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
            # flat = tf.reshape(pool, [-1, np.prod(pool.get_shape().as_list()[1:])])

            # dropout = tf.layers.dropout(inputs=flat, rate=0.4, training=train)
            # fc = tf.layers.dense(dropout, self.hidden_encoder_dim)
            # bn = batch_norm(fc, train=train)
            # h = lrelu(bn)

            # mu, sigma
            W_mean = self.weight_variable([self.hidden_encoder_dim, self.z_dim])
            b_mean = self.bias_variable([self.z_dim])
            z_mean = tf.matmul(h_encoder, W_mean) + b_mean

            W_logvar = self.weight_variable([self.hidden_encoder_dim, self.z_dim])
            b_logvar = self.bias_variable([self.z_dim])
            z_logvar = tf.matmul(h_encoder, W_logvar) + b_logvar

            return z_mean, z_logvar
            #######################################################
            #                   end of your code
            #######################################################


    def decoder(self, z, reuse=False, train=True):
        with tf.variable_scope("decoder", reuse=reuse):
            #######################################################
            # TODO: Define decoder network structure here. The size
            # of output should match the size of images. To make the
            # output pixel values in [0,1], add a sigmoid layer before
            # the output. Also use batch normalization layer after
            # deconv layer, and use 'train' argument to indicate the
            # mode of bn layer. Note that when sampling images using
            # trained model, you need to set train='False'.
            #######################################################
            W_decoder = self.weight_variable([self.z_dim, self.hidden_decoder_dim])
            b_decoder = self.bias_variable([self.hidden_decoder_dim])
            h_decoder = tf.nn.relu(tf.matmul(z, W_decoder) + b_decoder)

            W_out = self.weight_variable([self.hidden_decoder_dim, self.image_size**2])
            b_out = self.bias_variable([self.image_size**2])
            x_out = tf.matmul(h_decoder, W_out) + b_out
            
            x_hat = tf.nn.sigmoid(x_out)
            return x_hat
            #######################################################
            #                   end of your code
            #######################################################

    def build_model(self):
        #######################################################
        # TODO: In this build_model function, define inputs,
        # operations on inputs and loss of VAE. For input,
        # you need to define it as placeholders. Remember loss
        # term has two parts: reconstruction loss and KL divergence)
        # loss. Save the loss as self.loss. Use the
        # reparameterization trick to sample z.
        #######################################################
        self.x = tf.placeholder(tf.float32, shape=[None, self.image_size**2])
        # encoder
        z_mean, z_logvar = self.encoder(self.x)

        # reparameterize trick
        epsilon = tf.random_normal(tf.shape(z_logvar), name='epsilon')
        z_std = tf.exp(0.5 * z_logvar)
        z = z_mean + tf.multiply(z_std, epsilon)

        # decoder
        self.x_hat = self.decoder(z)

        KLD = -0.5 * tf.reduce_sum(1 + z_logvar - tf.pow(z_mean, 2) - tf.exp(z_logvar), reduction_indices=1)
        BCE = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_hat) + (1 - self.x) * tf.log(1e-10 + 1 - self.x_hat), 1)
        self.loss = KLD + BCE
        #######################################################
        #                   end of your code
        #######################################################
        self.saver = tf.train.Saver()

    def train(self, config):
        """Train VAE"""
        # load MNIST dataset
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        data = mnist.train.images
        data = data.astype(np.float32)
        data_len = data.shape[0]
        # data = np.reshape(data, [-1, 28, 28, 1])

        optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            tf.initialize_all_variables().run()

        start_time = time.time()
        counter = 1

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        sample_dir = os.path.join(config.sample_dir, config.dataset)
        if not os.path.exists(config.sample_dir):
            os.mkdir(config.sample_dir)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)

        for epoch in xrange(config.epoch):
            batch_idxs = min(data_len, config.train_size) // config.batch_size
            avg_cost = 0
            for idx in xrange(0, batch_idxs):
                counter += 1
                batch_images = data[idx*config.batch_size:(idx+1)*config.batch_size, :]
                #######################################################
                # TODO: Train your model here, print the loss term at
                # each training step to monitor the training process.
                # Print reconstructed images and sample images every
                # config.print_step steps. Sample z from standard normal
                # distribution for sampling images. You may use function
                # save_images in utils.py to save images.
                #######################################################
                _, loss = self.sess.run([optim, self.loss], feed_dict={self.x: batch_images})
                avg_cost += np.mean(loss) / batch_idxs
                #######################################################
                #                   end of your code
                #######################################################
                if np.mod(counter, 500) == 2 or (epoch == config.epoch-1 and idx == batch_idxs-1):
                    self.save(config.checkpoint_dir, counter)
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{}".format(avg_cost))
            if epoch % 10 == 0:
                images = self.sess.run(self.x_hat, feed_dict={self.x: batch_images})
                images = images.reshape((-1, 28, 28, 1))
                save_images(images, (28, 28), './img/digit{}.png'.format(epoch))
            
    def save(self, checkpoint_dir, step):
        model_name = "mnist.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
