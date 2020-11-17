import numpy as np
import tensorflow as tf

x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [1.0, 3.0, 5.0, 7.0]

w = tf.Variable(tf.random.normal([1], -100.0, 100.0))

for step in range(300):
    hypothesis = w * x
    cost = tf.reduce_mean(tf.square(hypothesis - y))

    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(w, x) - y, x))
    descent = w - tf.multiply(alpha, gradient)
    w.assign(descent)

    if step % 10 == 0:
        print("{:5} | {:10.4f} | {:10.6f}".format(step, cost.numpy(), w.numpy()[0]))

