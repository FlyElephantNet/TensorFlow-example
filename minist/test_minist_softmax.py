# -*- coding: UTF-8 -*-  
from PIL import Image
from numpy import *
import tensorflow as tf
import sys

if len(sys.argv) < 2 :
    print('argv must at least 2. you give '+str(len(sys.argv)))
    sys.exit()
filename = sys.argv[1]
im=Image.open(filename)
img = array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
data = img.reshape([1, 784])

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

saver = tf.train.Saver()
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    save_path = "minist_softmax.ckpt"
    saver.restore(sess, save_path)
    predictions = sess.run(y, feed_dict={x: data})
    print(predictions[0]);
