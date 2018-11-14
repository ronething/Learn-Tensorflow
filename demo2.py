# -*- coding:utf-8 _*-  
""" 
@author: ronething 
@file: demo2.py 
@time: 2018/11/14
@github: github.com/ronething 

Less is more.
"""

import tensorflow as tf

martix1 = tf.constant([[3, 3]])
martix2 = tf.constant([[2], [2]])
# np.dot(m1,m2)
product = tf.matmul(martix1, martix2)

# # method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
