
import matplotlib, numpy, pandas, scipy, sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import os
os.chdir("./MG/tf_data")

# regression using three inputs (x1, x2, x3)
# H(x1, x2, x3) = W1*x1 +  W2*x2 + W3*x3
# cost(W,b) = (1/m) * sum((H(x1, x2, x3) - y) ** 2)    #cost함수는 동일

# Hypothesis using matrix
# H(X) = X * W
## 여러개의 변수 쉽게 처리   [1, 10] [10, 1]  = [1, 1]
## 많은 인스턴스 한번에 처리   [n, 10] [10, 1] = [n, 1]
## 출력이 여러개인 경우도 처리  [n, 3] [3, 2] = [n, 2]

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1], name='weight1'))
w2 = tf.Variable(tf.random_normal([1], name='weight2'))
w3 = tf.Variable(tf.random_normal([1], name='weight3'))
b = tf.Variable(tf.random_normal([1], name='bias'))

hypothesis =  x1*w1 + x2*w2 + x3*w3 + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost : ", cost_val, "\nPrediction:\n", hy_val)


# 매트릭스 곱
x_data = [[73., 80., 75.], [93., 88., 93.],
          [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

X = tf.placeholder(tf.float32, shape=[None, 3])                # 쉐입 확인
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost : ", cost_val, "\nPrediction:\n", hy_val)


## 파일에다 적어두고 로딩하는 방법

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape=[None, 3])                # 쉐입 확인
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost : ", cost_val,
              "\nPrediction:\n", hy_val)

## 점수 예측
print("Your scor :", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
print("Your scor :", sess.run(hypothesis, feed_dict={X: [[60, 70, 110],[90, 100, 80]]}))

# Queue Runners
# 1) 파일들의 리스트 생성
# 2) 읽어올 reader를 정함
# 3) 어떻게 파싱할 것인가
# 4) 배치부분 만큼만 읽기

filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()                 # 텍스트 파일 읽기
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]]                 # float 타입으로 읽기
xy = tf.decode_csv(value, record_defaults=record_defaults)

## 배치로 읽기
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

## 세션 실행
sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val,
              "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)

