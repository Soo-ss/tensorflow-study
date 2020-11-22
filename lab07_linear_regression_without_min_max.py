import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(777)

xy = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

x_train = xy[:, 0:-1]
y_train = xy[:, [-1]]

plt.plot(x_train, "ro")
plt.plot(y_train)
# plt.show()

# 위 데이터를 기준으로 Linear Regression 모델을 만듬

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

W = tf.Variable(tf.random.normal((4, 1)), dtype=tf.float32)
b = tf.Variable(tf.random.normal((1,)), dtype=tf.float32)


def linearReg_fn(features):
    hypothesis = tf.matmul(features, W) + b
    return hypothesis


def loss_fn(hypothesis, features, labels):
    cost = tf.reduce_mean(tf.square(hypothesis - labels))
    return cost


optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)


def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(linearReg_fn(features), features, labels)
    return tape.gradient(loss_value, [W, b]), loss_value


# 학습

EPOCHS = 101

for step in range(EPOCHS):
    for features, labels in dataset:
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)
        hypo_value = linearReg_fn(features)
        grads, loss_value = grad(linearReg_fn(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
    print("Iter: {}, Loss: {:.4f}, Prediction: {}".format(step, loss_value, hypo_value))
