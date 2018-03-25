import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sess = tf.Session()


a = np.random.randint(0, 255, (7,5,5,3))
# print a
a_mean = np.mean(a, axis=(1,2), keepdims=True, dtype=np.int16)
# print a_mean
b = np.ones_like(a)
c = b * a_mean
print c[0,:,:,:]
print "--------------"
tt = tf.tile(tf.reduce_mean(a, axis=(1, 2), keep_dims=True), [1, 5, 5, 1])
print sess.run(tt - a_mean)[0,:,:,:]