
import matplotlib, numpy, pandas, scipy, sklearn
import pandas as pd
import matplotlib as plt
import numpy as np
import tensorflow as tf

import os
os.chdir("./MG/tf_data")

# SoftmaxRegression
# lr = WX
# lg = 1/(1+e**(-z)
# Multinomial classification
# n번 독립된 계산 --> Matrix multiplication(한번에 계산 가능!)
# 소프트 맥스
# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
# 각각이 0~1의 값(확률로 추출)
# 각각의 함이 1이 됨

# 소프트 맥스함수의 cost
# 크로스 엔트로피(Cross-entropy cost function)
## sum(L(i) * -log(hypothesis))
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# 경사하강법
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6],
          [1,6,6,6], [1,7,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]

X = tf.placeholder("float32", [None, 4])
Y = tf.placeholder("float32", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    all = sess.run(hypothesis, feed_dict={X: [[1,11,7,9],
                                              [1,3,4,3],
                                              [1,1,0,1]]})
    print(all, sess.run(tf.argmax(all, 1)))
    
# Test & one-hot encoding
# argmax를 활용

# Fancy Softmax Classification
# logits = tf.matmul(X, W)
# hypothesis = tf.nn.softmax(logits)
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
# cost = tf.reduce_mean(cost_i)

# Y = tf.placeholder(tf.int32, [None, 1])
# nb_classes = 7
# 1) Y_one_hot = tf.one_hot(Y, nb_classes)
# 2) Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

# 동물 종류 데이터셋 

xy = np.loadtxt("data-04-zoo.csv", delimiter=",", dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_predi = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predi, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print(step, loss, acc)
    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p==int(y), p, int(y)))









