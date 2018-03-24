from __future__ import division

import os
import math
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from ops import *
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

        self.obs = tf.placeholder(shape=[self.batch_size, self.image_size, self.image_size, 3], dtype=tf.float32)
        self.z = tf.placeholder(shape=[None, self.z_dim], dtype=tf.float32)
        self.cur_z = self.z
        self.loss = None

        self.build_model()

    def generator(self, inputs, reuse=False, is_training=True):
        ####################################################
        # Define the structure of generator, you may use the
        # generator structure of DCGAN. ops_backup.py defines some
        # layers that you may use.
        ####################################################
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.nn.relu(bn(linear(inputs, 1024, scope='g_fc1'), train=is_training, name='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 256 * 4 * 4, scope='g_fc2'), train=is_training, name='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 4, 4, 256])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 8, 8, 256], 4, 4, 2, 2, name='g_dc3'), train=is_training,
                   name='g_bn3'))
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 16, 16, 128], 4, 4, 2, 2, name='g_dc3-2'), train=is_training,
                   name='g_bn3-2'))
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 32, 32, 64], 4, 4, 2, 2, name='g_dc3-3'), train=is_training,
                   name='g_bn3-3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, self.image_size, self.image_size, 3], 4, 4, 2, 2,
                                         name='g_dc4'))

            return out


    def langevin_dynamics(self, z):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated z.
        ####################################################
        def body(i, z):
            y_hat = self.generator(z, reuse=True, is_training=True)
            BM = tf.random_normal(shape=tf.shape(z))
            # L = - 0.5 / self.delta ** 2 * tf.square(tf.norm((self.obs - y_hat)))
            L = - tf.reduce_mean(0.5 / self.delta ** 2 * tf.square(self.obs - y_hat), axis=0)
            gradient = tf.gradients(L, z)[0]
            z = z + self.delta * BM + 0.5 * self.delta ** 2 * (gradient - z)
            return (i+1, z)

        def cond(i, z):
            return i < self.sample_steps

        # y_hat_0 = self.generator(z, reuse=False, is_training=True)
        # BM_0 = tf.random_normal(shape=[self.batch_size, self.z_dim])
        # L_0 = - 0.5 / self.delta**2 * tf.square(tf.norm((self.obs - y_hat_0)))
        # gradient_0 = tf.gradients(L_0, z)[0]
        # z = z + self.delta * BM_0 + 0.5 * self.delta ** 2 * (gradient_0 - z)

        return tf.while_loop(cond, body, loop_vars=[tf.constant(0), z])[1]


    def build_model(self):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        y_hat = self.generator(self.z)
        # self.res = tf.reshape(self.obs - y_hat, shape=[self.batch_size, -1])
        # self.loss = 0.5 / self.delta**2 * tf.reduce_mean(tf.square(tf.norm(self.res, axis=1)))
        self.loss = 0.5 / self.delta**2 * tf.reduce_mean(tf.square(self.obs - y_hat), axis=0)
        self.loss_mean = tf.reduce_mean(self.loss)
        self.loss_sum = tf.summary.scalar("loss", self.loss_mean)

        self.cur_z = self.langevin_dynamics(self.z)

        # Test
        self.genImage = self.generator(self.z, reuse=True, is_training=False)
        self.saver = tf.train.Saver(max_to_keep=50)


    def train(self):
        # Prepare training data
        train_data = DataSet(self.data_path, image_size=self.image_size)
        train_data = train_data.to_range(-1, 1)

        # Training
        global_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.g_lr, global_steps, 100, 0.96, staircase=True)
        optim = tf.train.AdamOptimizer(learning_rate, beta1=self.beta1).minimize(self.loss, global_step=global_steps)

        # optim = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1).minimize(self.loss)
        num_batches = int(math.ceil(len(train_data) / self.batch_size))
        summary_op = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        counter = 1
        could_load, checkpoint_counter = self.load()
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

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

        cur_z = np.random.randn(self.batch_size, self.z_dim)

        # Q1
        loss_record = []
        for epoch in xrange(self.num_epochs):
            z = cur_z
            if np.mod(counter, self.log_step) == 0:
                self.save(counter)
                _, loss, loss_sum, cur_z, samples = \
                    self.sess.run([optim, self.loss_mean, self.loss_sum, self.cur_z, self.genImage],
                                  feed_dict={self.obs: train_data, self.z: z})

                save_images(samples, './{}/train_{:04d}.png'.format(self.sample_dir, counter))
                writer.add_summary(loss_sum, counter)

            else:
                _, loss, cur_z = self.sess.run([optim, self.loss_mean, self.cur_z],
                                               feed_dict={self.obs: train_data, self.z: z})

            loss_record.append(loss)
            print("loss = {}".format(loss))
            counter += 1

        print(" Finished training ...")
        plt.plot(loss_record)
        plt.show()

        # Q2
        rand_sampled_z = np.random.randn(self.batch_size, self.z_dim)
        randImages = self.sess.run(self.genImage, feed_dict={self.obs: train_data, self.z: rand_sampled_z})
        save_images(randImages, './{}/rand_sample_{:04d}.png'.format(self.sample_dir, counter))

        # Q3
        coord = np.linspace(-2,2,11)
        x, y = np.meshgrid(coord, coord)
        x, y = x.reshape(-1,), y.reshape(-1,)
        interp_z = np.column_stack((x, y))
        interpImages = np.zeros((121,64,64,3))
        for i in xrange(11):
            interpImages[i*11:(i+1)*11] = self.sess.run(self.genImage, feed_dict={self.obs: train_data, self.z: interp_z[i*11 : (i+1)*11]})
        save_images(interpImages, './{}/interpolated_sample_{:04d}.png'.format(self.sample_dir, counter))


    def save(self, step):
        model_name = "GenNet.model"
        checkpoint_dir = self.model_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = self.model_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
