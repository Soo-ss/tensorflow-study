import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# for tf >= 2.0
# tf.div => tf.compat.v1.div
# tf.log => tf.compat.v1.log
# tf.train.GradientDescentOptimizer => tf.optimizers.SGD

# tf.set_random_seed(seed) has changed to tf.random.set_seed(seed) in TensorFlow 2.
tf.random.set_seed(777)

# data 만들기
x_train = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 3.0], [5.0, 3.0], [6.0, 2.0]]
y_train = [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]]

x_test = [[5.0, 2.0]]
y_test = [[1.0]]


x1 = [x[0] for x in x_train]
x2 = [x[1] for x in x_train]

# plt
# colors = [int(y[0] % 3) for y in y_train]
# plt.scatter(x1, x2, c=colors, marker="^")
# plt.scatter(x_test[0][0], x_test[0][1], c="red")

# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()

# tensorflow eager
# 위 데이터들을 기준으로 가설의 검증을 통해 Logistic Classification 모델을 만듬
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

w = tf.Variable(tf.zeros([2, 1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# sigmoid함수를 가설로 선언
def logistic_regression(features):
    hypothesis = tf.compat.v1.div(1.0, 1.0 + tf.exp(tf.matmul(features, w) + b))
    return hypothesis


# 가설을 검증할 Cost함수를 정의함
def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(
        labels * tf.compat.v1.log(logistic_regression(features))
        + (1 - labels) * tf.compat.v1.log(1 - hypothesis)
    )
    return cost


optimizer = tf.optimizers.SGD(learning_rate=0.01)

# 추론한 값은 0.5를 기준으로 0과 1의 값을 리턴한다.
def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy


# GradientTape를 통해 경사값을 계산함
def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features), features, labels)
    return tape.gradient(loss_value, [w, b])


# Eager모드에서 학습을 실행함
EPOCHS = 1001

for step in range(EPOCHS):
    for features, labels in iter(dataset):
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [w, b]))
        if step % 100 == 0:
            print(
                "Iter: {}, Loss: {:.4f}".format(
                    step, loss_fn(logistic_regression(features), features, labels)
                )
            )
test_acc = accuracy_fn(logistic_regression(x_test), y_test)
print("Testset Accuracy: {:.4f}".format(test_acc))

