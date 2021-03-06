
import matplotlib, numpy, pandas, scipy, sklearn
import pandas as pd
import matplotlib as plt
import numpy as np
import tensorflow as tf

import os
os.chdir("./MG/tf_data")

# Binary Classification
## 0,1 encoding
## sigmoid 함수(로지스틱 함수)
# g(z) = 1 / (1 + e**(-z))
# H(x) = 1 / (1 + e**(-W*x))
# cost(W) = (1/m) * sum( c(H(x),y) )
# c(H(x), y) = -log(H(x))   : y = 1
# c(H(x), y) = -log(1 - H(x))   : y = 0
## c(H(x), y) = ylog(H(x)) - (1 - y)log(1 - H(x))
# cost(W) = -(1/m) * sum(ylog(H(x)) - (1 - y)log(1 - H(x)))

# 비용 함수
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis)))

# 최소화는 동일
# W := W - a * (∂/∂W)cost(W)

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data =[[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)                      ## tf.cast()  True나 False값은 1과 0으로 치환
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nhypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


## 당뇨병 분류 예측
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = {X: x_data, Y: y_data}
    for step in range(10001):
        sess.run(train, feed_dict= feed)
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict = feed))
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nhypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)






