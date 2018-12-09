
import matplotlib, numpy, pandas, scipy, sklearn
import pandas as pd
import matplotlib as plt
import numpy as np
import tensorflow as tf

import os
os.chdir("./MG/tf_data")

# 학습률을 잘 정하는 것이 중요하다!
# Large learning_rate
# overshooting : 비용값이 뛰어나가는 현상(학습률이 클 때)
# Small learning_rate
# takes too long, stops at local minimum

# 전처리가 되었는가?
# Data preprocessing for gradient descent
# original data -> zero-centered data
# original data -> normalized data
# Standardization
# X_std[:0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()

# 과대적합인 경우
# Overfitting
# Solutions for overfitting
# 1) More training data!
# 2) Reduce the number of features
# 3) Regularization(일반화)
## Regularization(일반화)

## regularization strength(λ) = 0.001
# l2reg = 0.001 * tf.reduce_sum(tf.square(W))


# 평가 방법
# Performance evaluation: is this good?
# Training and Test sets
# /            Training              / Validation /     Testing     /

# 온라인 학습(<-> 배치학습)
# MINIST Dataset
# Accuracy

### Training and Test datasets


x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2, 5],
          [1, 6, 6],[1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0],
          [1, 0, 0], [1, 0, 0]]

x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.random_normal([3,3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, train],
         feed_dict={X: x_data, Y: y_data})
        if step % 10 == 0:
            print(step, cost_val, W_val)
    print("Prediction: ", sess.run(prediction, feed_dict={X: x_test}))
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))


## prediction 확인할 때 test 데이터셋 사용!


# Learning rate: NaN!

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.random_normal([3,3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(1.5).minimize(cost)           # Big learning rate

prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, train],
         feed_dict={X: x_data, Y: y_data})
        if step % 10 == 0:
            print(step, cost_val, W_val)
    print("Prediction: ", sess.run(prediction, feed_dict={X: x_test}))
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

## nan : 무한대로 학습 포기  <- learning_rate를 낮춘다

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.random_normal([3,3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(1e-10).minimize(cost)           # Small learning rate

prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, train],
         feed_dict={X: x_data, Y: y_data})
        if step % 10 == 0:
            print(step, cost_val, W_val)
    print("Prediction: ", sess.run(prediction, feed_dict={X: x_test}))
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

## 수백번을 하여도 오차값이 줄어들지 않는다(작은 홀에 빠짐!) <- learning_rate를 높인다


# Non-normalized inputs

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])


# 스케일을 안했을 경우: Nan
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(1e-5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run(
    [cost, hypothesis, train], feed_dict = {X: x_data, Y: y_data})
    if step % 100 == 0:
        print(step, cost_val, hy_val)


# 정규화 min-max scale
xy = MinMaxScaler(xy)

from sklearn.preprocessing import MinMaxScaler

xy = MinMaxScaler().fit_transform(xy)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(1e-5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run(
    [cost, hypothesis, train], feed_dict = {X: x_data, Y: y_data})
    if step % 100 == 0:
        print(step, cost_val, hy_val)

## 데이터가 굉장히 크거나 들쭉날쭉할 때는 반듯이 Normalized inputs


