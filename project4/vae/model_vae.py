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
import numpy as np

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

        self.z_dim = z_dim
        self.c_dim = c_dim

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.z1_shape = None
        self.z2_shape = None
        self.z2_flat_shape = None
        self.z3_shape = None

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
            z = lrelu(conv2d(image, 16, name="e_z1"), name="e_z1_lrelu")
            self.z1_shape = z.shape
            z = lrelu(batch_norm(conv2d(z, 32, name="e_z2"), train=train, name="e_z2_bn"), name="e_z2_lrelu")
            self.z2_shape = z.shape
            z = tf.reshape(z, [self.batch_size, -1])
            self.z2_flat_shape = z.shape
            z = lrelu(batch_norm(linear(z, 512, 'e_z3'), train=train, name="e_z3_bn"), name="e_z3_lrelu")
            self.z3_shape = z.shape

            w_mean = linear(z, self.z_dim, "w_mean")
            w_stddev = tf.nn.softplus(linear(z, self.z_dim, "w_stddev")) + 1e-6

            return w_mean, w_stddev
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
            x_hat = lrelu(batch_norm(linear(z, self.z3_shape[1], "x_hat_1"),
                                     train=train, name="x_hat_1_bn"),
                          name="x_hat_1_lrelu")
            x_hat = lrelu(batch_norm(linear(x_hat, self.z2_flat_shape[1], "x_hat_2"),
                                     train=train, name="x_hat_2_bn"),
                          name="x_hat_2_lrelu")
            x_hat = tf.reshape(x_hat, self.z2_shape)
            x_hat = lrelu(batch_norm(deconv2d(x_hat, self.z1_shape, name="x_hat_3"),
                                     train=train, name="x_hat_3_bn"),
                          name="x_hat_3_lrelu")
            x_hat = tf.nn.sigmoid(deconv2d(x_hat, [self.batch_size, self.image_size, self.image_size, 1],
                                           name="x_hat_4"))

            return x_hat
            #######################################################
            #                   end of your code
            #######################################################

    def build_model(self):
        #######################################################
        # TODO: In this build_model function, define inputs,
        # operations on inputs and loss of VAE. For input,
        # you need to define it as placeholders. Remember loss
        # term has two parts: reconstruction loss and KL divergence
        # loss. Save the loss as self.loss. Use the
        # reparameterization trick to sample z.
        #######################################################
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_size, self.image_size, 1])

        # Encoder
        z_mean, z_stddev = self.encoder(self.x, train=True)

        # Reparameterization
        epsilon = tf.random_uniform([self.batch_size, self.z_dim], 0, 1, name="epsilon")
        z = z_mean + z_stddev * epsilon

        # Decoder
        self.x_hat = self.decoder(z, train=True)
        self.x_hat = tf.clip_by_value(self.x_hat, 1e-8, 1 - 1e-8)

        # Compute Loss
        BCE = -tf.reduce_sum(self.x * tf.log(self.x_hat) + (1 - self.x) * tf.log(1 - self.x_hat), [1, 2, 3])
        KLD = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev) + 1e-8) - 1, 1)
        self.loss = tf.reduce_mean(BCE + KLD)
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
        data = np.reshape(data, [-1, 28, 28, 1])

        optim = tf.train.AdamOptimizer(config.learning_rate * 10, beta1=config.beta1).minimize(self.loss)
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
            avg_loss = 0
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
                avg_loss += loss / batch_idxs
                #######################################################
                #                   end of your code
                #######################################################
                if np.mod(counter, 500) == 2 or (epoch == config.epoch-1 and idx == batch_idxs-1):
                    self.save(config.checkpoint_dir, counter)

            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{}".format(avg_loss))
            if epoch % 10 == 0:
                images = self.sess.run(self.x_hat, feed_dict={self.x: batch_images})
                save_images(images, (28, 28), './img3/digit{}.png'.format(epoch))
            
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
