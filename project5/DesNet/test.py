import tensorflow as tf
import numpy as np

a = np.arange(0,105).reshape(7,3,5)
# print a
a_mean = np.mean(a, axis=(1,2), keepdims=True)
print a_mean.shape
b = np.ones_like(a)
print b
print a_mean
print b * a_mean