# -*- coding:utf-8 _*-  
""" 
@author: ronething 
@file: demo4.py 
@time: 2018/11/14
@github: github.com/ronething 

Less is more.
"""

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
