import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

a = np.arange(0,420).reshape(7,4,5,3)
# print a
a_mean = np.mean(a, axis=(1,2), keepdims=True)
print a_mean.shape
# print a_mean
b = np.ones_like(a)
c = b * a_mean
print c[0,:,:,:]
print c.shape
