import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(777)


xy = np.loadtxt(
    "data-04-zoo.csv", delimiter=",", dtype=np.float32
)  # tf1.13.1에서는 np.int32, 이전에는 np.float32
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7  # 0 ~ 6

X = tf.compat.v1.placeholder(tf.float32, [None, 16])
Y = tf.compat.v1.placeholder(tf.int32, [None, 1])

# Make Y data as onehot shape
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

print(x_data.shape, Y_one_hot.shape)


# Weight and bias setting
W = tf.Variable(tf.random.normal([16, nb_classes]), name="weight")
b = tf.Variable(tf.random.normal([nb_classes]), name="bias")
variables = [W, b]

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
def logit_fn(X):
    return tf.matmul(X, W) + b


def hypothesis(X):
    return tf.nn.softmax(logit_fn(X))


def cost_fn(X, Y):
    logits = logit_fn(X)
    cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
    cost = tf.reduce_mean(cost_i)

    return cost


def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        loss = cost_fn(X, Y)
        grads = tape.gradient(loss, variables)
        return grads


def prediction(X, Y):
    pred = tf.argmax(hypothesis(X), 1)
    correct_prediction = tf.equal(pred, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def fit(X, Y, epochs=1000, verbose=100):
    optimizer = tf.optimizers.SGD(learning_rate=0.1)

    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i == 0) | ((i + 1) % verbose == 0):
            # print('Loss at epoch %d: %f' %(i+1, cost_fn(X, Y).numpy()))
            acc = prediction(X, Y).numpy()
            loss = cost_fn(X, Y).numpy()
            print("Steps: {} Loss: {}, Acc: {}".format(i + 1, loss, acc))


fit(x_data, Y_one_hot)
