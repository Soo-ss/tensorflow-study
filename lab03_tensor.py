import numpy as np
import tensorflow as tf

x = np.array([1, 2, 3])
y = np.array([1, 2, 3])


def cost_func(w, x, y):
    hypothesis = x * w
    return tf.reduce_mean(tf.square(hypothesis - y))


w_values = np.linspace(-3, 5, num=15)
cost_values = []

for feed_w in w_values:
    curr_cost = cost_func(feed_w, x, y)
    cost_values.append(curr_cost)
    print("{:6.3f} | {:10.5f}".format(feed_w, curr_cost))

