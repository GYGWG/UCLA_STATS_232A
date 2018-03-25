# -*- coding: utf-8 -*-
import os
from glob import glob
import time
import tensorflow as tf
import numpy as np

from ops import *
from datasets import *

import matplotlib.pyplot as plt

class DesNet(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.batch_size = flags.batch_size

        self.data_path = os.path.join('./Image', flags.dataset_name)
        self.output_dir = flags.output_dir

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.sample_dir = os.path.join(self.output_dir, 'sample')
        self.model_dir = os.path.join(self.output_dir, 'checkpoints')

        # For training
        self.obs = tf.placeholder(
            shape=[None, flags.image_size, flags.image_size, 3],
            dtype=tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        # For Langevin
        self.energy = None
        self.df_dy = None
        self.mean_Y = None
        self.lang_out = None

        # For Loss
        self.desc_out = None
        self.loss = None

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.build_model(flags)

    def descriptor(self, inputs, is_training=True, reuse=False):
        ####################################################
        # Define network structure for descriptor.
        # Recommended structure:
        # conv1: channel 64 kernel 4*4 stride 2
        # conv2: channel 128 kernel 2*2 stride 1
        # fc: channel output 1
        # conv1 - bn - relu - conv2 - bn - relu -fc
        ####################################################
        with tf.variable_scope("descriptor", reuse=tf.AUTO_REUSE):
            # Conv Layer 1: output size is 32 * 32 * 64
            conv_h1 = conv2d(inputs, 64, 4, 4, 2, 2, name="conv_h1")
            bn_h1 = bn(conv_h1, train=is_training, name="bn_h1")
            out_h1 = lrelu(bn_h1)
                
            # Conv Layer 2: output size is 32 * 32 * 128
            conv_h2 = conv2d(out_h1, 128, 2, 2, 1, 1, name="conv_h2")
            bn_h2 = bn(conv_h2, train=is_training, name="bn_h2")
            out_h2 = lrelu(bn_h2)
                
            # FC Layer
            hl_out_flatten = tf.contrib.layers.flatten(out_h2)
            out = linear(hl_out_flatten, 1, scope="out")

            return out

    def Langevin_sampling(self, samples, flags):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated samples.
        ####################################################
        i = tf.constant(0)

        # Get gradient of energy
        df_dy = tf.gradients(self.energy, samples)[0]
        self.df_dy = df_dy

        def cond(i, _samples):
            return tf.less(i, flags.T)

        def body(i, samples):
            U = flags.delta * tf.random_normal([self.batch_size, 64, 64, 3])
            # Update
            samples = samples - (0.5 * flags.delta ** 2) * df_dy + U
            i = tf.add(i, 1)
            return i, samples

        res = tf.while_loop(cond, body, [i, samples], shape_invariants=[i.shape, samples.shape])

        return res[1]

    def build_model(self, flags):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        # mean image -- TODO is it img - mean?
        self.mean_Y = tf.tile(tf.reduce_mean(self.obs, axis=(1, 2), keep_dims=True), [1, 64, 64, 1])
        self.desc_out = self.descriptor(self.mean_Y)
        self.energy = 1 / (2 * flags.ref_sig) * l2(self.mean_Y) - self.desc_out

        # Langevin Sampling
        self.lang_out = self.Langevin_sampling(self.mean_Y, flags)

        # Loss function
        self.loss = - tf.reduce_mean(self.descriptor(self.obs)) + tf.reduce_mean(self.descriptor(self.lang_out))

    def train(self, flags):
        # Prepare training data, scale is [0, 255]
        train_data = DataSet(self.data_path, image_size=flags.image_size)
        train_data = train_data.to_range(0, 255)
        summary_op = tf.summary.merge_all()

        # Optimizer setup
        optim = tf.train.AdamOptimizer(flags.learning_rate, beta1=flags.beta1).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50)
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        #self.sess.graph.finalize()

        print(" Start training ...")

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # synthesized images in self.sample_dir,
        # loss in self.log_dir (using writer).
        ####################################################
        loss_history = []
        for epoch in range(flags.epoch):
            _, lang_out, loss = self.sess.run([optim, self.lang_out, self.loss], feed_dict={self.obs: train_data,
                                                                                            self.is_training: True})
            loss_history.append(loss)

            print("Epoch: {}, Loss: {}".format(epoch, loss))

            if (epoch + 1) % 100 == 0:
                save_images(lang_out, "synthesized/epoch" + str(epoch+1) + ".jpg")

        # (2) Pring Loss over time
        plt.title("Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(np.arange(1000), loss_history)
        plt.show()
