import tensorflow as tf
import collections
import numpy as np
from collections import deque

sess = tf.Session()

i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i])
print sess.run(r)

Pair = collections.namedtuple('Pair', 'j, k')
ijk_0 = (tf.constant(0), Pair(tf.constant(1), tf.constant(2)))
c = lambda i, p: i < 10
b = lambda i, p: (i + 1, Pair((p.j + p.k), (p.j - p.k)))
ijk_final = tf.while_loop(c, b, ijk_0)
print sess.run(ijk_final)



class test:
    def __init__(self):
        self.a = 0
        self.b = tf.constant(20)

    def while_loop(self):
        def body(i, z):
            i += 1
            z += 2
            return (i, z)

        def cond(i, z):
            return i < self.b

        return tf.while_loop(cond, body, [self.a, self.a])

tt = test()
print sess.run(tt.while_loop())

a = np.ones((11,4))
b = np.zeros((4,4))
c = np.row_stack((a,b))
print c

dq = deque(maxlen=5)
for i in xrange(9):
    dq.append(i)

print dq
print dq[3]

def test(d):
    i = np.random.randint(0, 4)
    return d[i]

tt = tf.range(1,10)
for i in xrange(3):
    print(sess.run(test(tt)))