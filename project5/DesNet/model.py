# -*- coding: utf-8 -*-
import os
from glob import glob
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ops import *
from datasets import *


class DesNet(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.batch_size = flags.batch_size
        self.imgSize = flags.image_size

        self.data_path = os.path.join('./Image', flags.dataset_name)
        self.output_dir = flags.output_dir
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

        self.img = tf.placeholder(shape=(self.batch_size, self.imgSize, self.imgSize, 3), dtype=tf.float32)
        self.meanImg = tf.placeholder(shape=(self.batch_size, self.imgSize, self.imgSize, 3), dtype=tf.float32)

        self.build_model(flags)

    def descriptor(self, inputs, is_training=True, reuse=False):
        with tf.variable_scope("descriptor", reuse=reuse) as scope:
            ####################################################
            # Define network structure for descriptor.
            # Recommended structure:
            # conv1: channel 64 kernel 4*4 stride 2
            # conv2: channel 128 kernel 2*2 stride 1
            # fc: channel output 1
            # conv1 - bn - relu - conv2 - bn - relu -fc
            ####################################################
            # net = lrelu(bn(conv2d(inputs, 32, 4, 4, 2, 2, name='d_conv1'), train=is_training, name='d_bn1'))
            # net = lrelu(bn(conv2d(net, 64, 4, 4, 2, 2, name='d_conv2'), train=is_training, name='d_bn2'))
            # net = tf.reshape(net, [self.batch_size, -1])
            # net = lrelu(bn(linear(net, 1024, scope='d_fc1'), train=is_training, name='d_bn3'))
            net = lrelu(bn(conv2d(inputs, 64, 4, 4, 2, 2, name='d_conv1'), train=is_training, name='d_bn1'))
            net = lrelu(bn(conv2d(net, 128, 2, 2, 1, 1, name='d_conv2'), train=is_training, name='d_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            return linear(net, 1, 'd_fc3')


    def Langevin_sampling(self, samples, flags, train=True, reuse=True):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated samples.
        ####################################################
        def body(i, img):
            # E = 0.5 / flags.ref_sig**2 * tf.square(tf.norm(img)) - self.descriptor(img, is_training=True, reuse=True)
            # img = img - 0.5 * flags.delta**2 * tf.gradients(E, img)[0] + tf.random_normal(shape=img.shape)
            BM = tf.random_normal(shape=tf.shape(img), name='noise')
            y_hat = self.descriptor(img, is_training=train, reuse=reuse)
            grad = tf.gradients(y_hat, img, name='grad_des')[0]
            img = img - 0.5 * flags.delta ** 2 * (img / flags.ref_sig ** 2 - grad) + flags.delta * BM
            return i + 1, img

        def cond(i, img):
            return i < flags.T

        return tf.while_loop(cond, body, loop_vars=(tf.constant(1), samples))[1]

    def build_model(self, flags):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        # Real Data
        img_scoreFun = self.descriptor(self.img)
        # img = tf.reshape(self.img, (self.batch_size, -1))
        # norm = tf.norm(img, axis=1)
        # E_img = 0.5 / flags.ref_sig**2 * tf.square(tf.reshape(norm, (-1, 1))) - img_scoreFun

        # Dream data
        # E_synImg = self.dreamingData(self.meanImg, flags)
        synImg = self.Langevin_sampling(self.meanImg, flags)
        synImg_scoreFun = self.descriptor(synImg, reuse=True, is_training=True)

        # loss
        self.loss = tf.subtract(tf.reduce_mean(synImg_scoreFun), tf.reduce_mean(img_scoreFun))
        # self.loss = tf.subtract(synImg_scoreFun, img_scoreFun)
        # self.loss_mean = tf.reduce_mean(self.loss)
        self.loss_sum = tf.summary.scalar("loss", self.loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]

        # test
        self.synImg = self.Langevin_sampling(self.meanImg, flags, train=False)

        self.saver = tf.train.Saver(max_to_keep=50)


    def train(self, flags):
        # Prepare training data, scale is [0, 255]
        train_data = DataSet(self.data_path, image_size=flags.image_size)
        train_data = train_data.to_range(0, 255)
        # data_mean = np.mean(train_data, axis=(1,2,3),keepdims=True)
        data_mean = np.mean(train_data, axis=(1, 2), keepdims=True)

        train_meanData = np.ones_like(train_data) * data_mean

        # Exp-decay learning rate
        counter = 0
        # global_steps = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(flags.learning_rate, global_steps, 50, 0.96, staircase=True)
        # optim = tf.train.AdamOptimizer(learning_rate, beta1=flags.beta1).minimize(self.loss, global_step=global_steps, var_list=self.d_vars)

        optim = tf.train.AdamOptimizer(flags.learning_rate, beta1=flags.beta1).minimize(self.loss, var_list=self.d_vars)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        could_load, checkpoint_counter = self.load()
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.sess.graph.finalize()

        print(" Start training ...")


        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # synthesized images in self.sample_dir,
        # loss in self.log_dir (using writer).
        ####################################################
        loss_record = []
        for epoch in xrange(flags.epoch):
            if np.mod(counter, flags.log_steps) == 0:
                _, loss, loss_sum, synImg = \
                    self.sess.run([optim, self.loss, self.loss_sum, self.synImg],
                                  feed_dict={self.img: train_data, self.meanImg: train_meanData})

                save_images(synImg, './{}/train_{:00d}.png'.format(self.sample_dir, counter))
                self.writer.add_summary(loss_sum, counter)
                self.save(counter)

            else:
                _, loss, loss_sum = \
                    self.sess.run([optim, self.loss, self.loss_sum],
                                  feed_dict={self.img: train_data, self.meanImg: train_meanData})

            loss_record.append(loss)
            print("loss = {}".format(loss))
            counter += 1

        print(" Finished training ...")
        plt.plot(loss_record)
        plt.show()


    def dreamingData(self, meanImg, flags):
        def body(i, E_collection):
            synImg = self.Langevin_sampling(meanImg, flags)
            synImg_scoreFun = self.descriptor(synImg, reuse=True, is_training=True)
            synImg = tf.reshape(synImg, (self.batch_size, -1))
            norm = tf.norm(synImg, axis=1)
            E_synImg = 0.5 / flags.ref_sig ** 2 * tf.square(tf.reshape(norm, (-1, 1))) - synImg_scoreFun
            E_collection = tf.concat((E_collection, E_synImg), axis=0)
            return i+1, E_collection

        def cond(i, E_collection):
            return i < flags.n

        E_collection = tf.constant(0.0, shape=(1,1))
        ind = tf.constant(1)
        return tf.while_loop(cond, body, loop_vars=(ind, E_collection),
                             shape_invariants=(ind.shape,
                                               tf.TensorShape([None, 1])))[1][1:]


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
