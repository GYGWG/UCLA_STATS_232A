import tensorflow as tf
import collections

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


z1 = tf.placeholder(shape=[2], dtype=tf.float32)
z2 = z1
z2 = z2 + tf.random_normal(z2.shape)

print sess.run([z1, z2], feed_dict={z1: [1, 1]})


class test:
    def __init__(self):
        self.a = 0
        self.b = tf.constant(20)

    def while_loop(self):
        def body(i, z):
            return [i+1, z+2]

        def cond(i, z):
            return i < self.b

        return tf.while_loop(cond, body, [self.a, self.a])

tt = test()
print sess.run(tt.while_loop())

a = tf.ones(shape=[2,3,4,5])
b = tf.norm(a, axis=[1,2,3])
print sess.run(b)

