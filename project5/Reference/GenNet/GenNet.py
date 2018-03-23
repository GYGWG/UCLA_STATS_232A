from __future__ import division

import os
import math
import numpy as np
import tensorflow as tf

from ops import leaky_relu
from datasets import *


class GenNet(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.delta = config.delta
        self.sigma = config.sigma
        self.sample_steps = config.sample_steps
        self.z_dim = config.z_dim

        self.num_epochs = config.num_epochs
        self.data_path = os.path.join(config.data_path, config.category)
        self.log_step = config.log_step
        self.output_dir = os.path.join(config.output_dir, config.category)

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.sample_dir = os.path.join(self.output_dir, 'sample')
        self.model_dir = os.path.join(self.output_dir, 'checkpoints')

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.obs = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32)
        self.z = tf.placeholder(shape=[None, self.z_dim], dtype=tf.float32)
        self.build_model()

    def generator(self, inputs, reuse=False, is_training=True):
        ####################################################
        # Define the structure of generator, you may use the
        # generator structure of DCGAN. ops.py defines some
        # layers that you may use.
        ####################################################
        with tf.variable_scope('gen') as scope:

    def langevin_dynamics(self, z):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated z.
        ####################################################
        def cond(i, z):
            return tf.less(i, self.sample_steps)

        def body(i, z):
            noise = tf.random_normal(shape=tf.shape(z), name='noise')
            gen_res = self.generator(z, reuse=True)
            gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma * self.sigma) * tf.square(self.obs - gen_res), axis=0)
            grad = tf.gradients(gen_loss, z, name='grad_des')[0]
            z = z - 0.5 * self.delta * self.delta * (z + grad)
            z = z + self.delta * noise
            return tf.add(i, 1), z

        i = tf.constant(0)
        i, z = tf.while_loop(cond, body, [i, z_arg])
        return z

    def build_model(self):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        g_res = self.generator(self.z)
        self.gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma * self.sigma) * tf.square(self.obs - g_res), axis=0)

        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        self.infer_op = self.langevin_dynamics(self.z)
        self.train_op = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1).minimize(gen_loss, var_list=self.g_vars)

    def train(self):
        # Prepare training data
        train_data = DataSet(self.data_path, image_size=self.image_size)
        train_data = train_data.to_range(-1, 1)

        num_batches = int(math.ceil(len(train_data) / self.batch_size))
        summary_op = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50)
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.sess.graph.finalize()

        print('Start training ...')

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # reconstructed images and synthesized images in
        # self.sample_dir, loss in self.log_dir (using writer).
        ####################################################
