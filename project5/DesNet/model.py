# -*- coding: utf-8 -*-
import os
from glob import glob
import time
import tensorflow as tf
import numpy as np
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

        self.img = tf.placeholder(shape=(None, self.imgSize, self.imgSize, 3), dtype=tf.float32)
        self.meanImg = tf.placeholder(shape=(None, self.imgSize, self.imgSize, 3), dtype=tf.float32)

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
            net = lrelu(conv2d(inputs, 64, 4, 4, 2, 2, name='en_conv1'))
            net = lrelu(bn(conv2d(net, 128, 2, 2, 1, 1, name='en_conv2'), train=is_training, name='en_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            return linear(net, 1, 'en_fc3')
            # gaussian_params = linear(net, 1, scope='en_fc4')


    def Langevin_sampling(self, samples, flags):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated samples.
        ####################################################
        def body_inner(i, img):
            E = 0.5 / flags.ref_sig**2 * tf.square(tf.norm(img)) - self.descriptor(img, is_training=True, reuse=True)
            img = img - 0.5 * flags.delta**2 * tf.gradients(E, img) + tf.random_normal(shape=img.shape)
            return i + 1, img

        def cond_inner(i, img):
            return i < flags.T

        def body_outer(i, c):
            s = tf.while_loop(cond_inner, body_inner, loop_vars=[tf.constant(1), samples])[1]
            c = tf.concat((c, s), axis=0)
            return i + 1, c

        def cond_outer(i, c):
            return i < flags.n

        ind = tf.constant(1)
        collection = tf.while_loop(cond_inner, body_inner, loop_vars=(ind, samples))[1]
        return tf.while_loop(cond_outer, body_outer, loop_vars=(ind, collection),
                             shape_invariants=(ind.shape,
                                               tf.TensorShape([None, flags.image_size, flags.image_size, 3])))[1]


    def build_model(self, flags):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        # Real Data
        img_scoreFun = self.descriptor(self.img, reuse=False, is_training=True)
        img = self.img.reshape(self.batch_size, -1)
        E_img = 0.5 / flags.ref_sig**2 * tf.norm(img, axis=1) - img_scoreFun

        # Dream data
        self.synImg = self.Langevin_sampling(self.meanImg, flags)
        synImg_scoreFun = self.descriptor(self.synImg, reuse=True, is_training=True)
        synImg = self.synImg.reshape(self.batch_size * flags.n, -1)
        E_synImg = 0.5 / flags.ref_sig**2 * tf.norm(synImg, axis=1) - synImg_scoreFun

        # loss
        self.loss = tf.reduce_mean(E_synImg) - tf.reduce_mean(E_img)
        self.loss_sum = tf.summary.scalar("loss", self.loss)

        self.saver = tf.train.Saver(max_to_keep=50)


    def train(self, flags):
        # Prepare training data, scale is [0, 255]
        train_data = DataSet(self.data_path, image_size=flags.image_size)
        train_data = train_data.to_range(0, 255)
        data_mean = np.mean(train_data, axis=(1,2,3),keepdims=True)
        train_meanData = np.ones_like(train_data) * data_mean
        optim = tf.train.AdamOptimizer(flags.learning_rate, beta1=flags.beta1).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        counter = 1
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

        for epoch in xrange(flags.epoch):
            if np.mod(counter, flags.log_steps) == 0:
                _, loss, loss_sum, synImg = \
                    self.sess.run([optim, self.loss, self.loss_sum, self.synImg],
                                  feed_dict={self.img: train_data, self.meanImg: train_meanData})

                save_images(synImg, './{}/train_{:04d}.png'.format(self.sample_dir, counter))
                self.writer.add_summary(loss_sum, counter)

            else:
                _, loss, loss_sum = \
                    self.sess.run([optim, self.loss, self.loss_sum],
                                  feed_dict={self.img: train_data, self.meanImg: train_meanData})

            print("loss = {}".format(loss))
            counter += 1

        print(" Finished training ...")











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
