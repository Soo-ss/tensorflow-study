import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

tf.random.set_seed(777)

x_train = [
    [1, 2, 1],
    [1, 3, 2],
    [1, 3, 4],
    [1, 5, 5],
    [1, 7, 5],
    [1, 2, 5],
    [1, 6, 6],
    [1, 7, 7],
]

y_train = [
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

x1 = [x[0] for x in x_train]
x2 = [x[1] for x in x_train]
x3 = [x[2] for x in x_train]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x1, x2, x3, c=y_train, marker="^")

ax.scatter(x_test[0][0], x_test[0][1], x_test[0][2], c="black", marker="^")
ax.scatter(x_test[1][0], x_test[1][1], x_test[1][2], c="black", marker="^")
ax.scatter(x_test[2][0], x_test[2][1], x_test[2][2], c="black", marker="^")


ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")

# plt.show()

# Tensorflow Eager
# 위 데이터를 기준으로 Learning Rate값과 평가 모델을 만듬

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

# Softmax Classification
W = tf.Variable(tf.random.normal((3, 3)))
b = tf.Variable(tf.random.normal((3,)))

# Softmax 함수를 가설로 선언한다.
def softmax_fn(features):
    return tf.nn.softmax(tf.matmul(features, W) + b)


# 가설을 검증할 Cost함수 정의
def loss_fn(hypothesis, features, labels):
    return tf.reduce_mean(-tf.reduce_sum(labels * tf.math.log(hypothesis), axis=1))


# Learning Rate값을 조정하기 위한 Learning Decay 설정
"""
1. starter_learning_rate : 최초 학습시 사용될 learning rate (0.1로 설정하여 0.96씩 감소하는지 확인)
2. global_step : 현재 학습 횟수
3. 1000 : 곱할 횟수 정의 (1000번에 마다 적용)
4. 0.96 : 기존 learning에 곱할 값
5. 적용유무 decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
"""
is_decay = True
starter_learning_rate = 0.1

if is_decay:
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=starter_learning_rate,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate)
else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=starter_learning_rate)


def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(softmax_fn(features), features, labels)
    return tape.gradient(loss_value, [W, b])


def accuracy_fn(hypothesis, labels):
    prediction = tf.argmax(hypothesis, 1)
    is_correct = tf.equal(prediction, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    return accuracy


# Tensorflow 학습 진행
EPOCHS = 1001

for step in range(EPOCHS):
    for features, labels in iter(dataset):
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)
        grads = grad(softmax_fn(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        if step % 100 == 0:
            print(
                "Iter: {}, Loss: {:.4f}".format(
                    step, loss_fn(softmax_fn(features), features, labels)
                )
            )
x_test = tf.cast(x_test, tf.float32)
y_test = tf.cast(y_test, tf.float32)
test_acc = accuracy_fn(softmax_fn(x_test), y_test)
print("Testset Accuracy: {:.4f}".format(test_acc))
