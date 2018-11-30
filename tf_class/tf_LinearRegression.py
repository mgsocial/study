
import matplotlib, numpy, pandas, scipy, sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import os
os.chdir("./MG/tf_data")

## Hypothesis and Cost
# H(x) = W * x + b
# cost(W, b) = (1/m) * sum((H(x) - y)^2)

## Simplified hypothesis
# H(x) = W * x
# cost(W) = (1/m) * sum((W * x - y)^2)

## 경사하강법(Gradient decent algorithm) 최소화
# cost(W) = (1/2m) * sum((W * x - y)^2)    # Convex function(밥그릇 모양)임을 확인한다
# W := W - a * (∂/∂W)cost(W)             # W를 미분한 값을 뺀다
# W := W - a * (1/m) * sum(W * x - y) * x

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float64)
hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []
for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)


## 그래디언트 디센트 구현
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

## Minimize : 그래디언트 디센트 Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(-3.0)
hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(101):
    if step % 10 == 0:
        print(step, sess.run(W))
    sess.run(train)

# 추가 내용 : compute_gradient and apply_gradient

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(5.)
hypothesis = X * W

# Manual gradient
gradient = tf.reduce_mean((W * X - Y) * X) * 2       # 미분의 수식

# cost/Loss funtion
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Get gradients
gvs = optimizer.compute_gradients(cost)               # gvs를 아무런 수정도 하지 않음

# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)

### 텐서플로에서 계산해주는 그레디언트 값의 거의 차이는 없다!!
