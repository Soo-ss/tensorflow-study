import numpy as np
import tensorflow as tf

data = np.array(
    [
        [73.0, 80.0, 75.0, 152.0],
        [93.0, 88.0, 93.0, 185.0],
        [89.0, 91.0, 90.0, 180.0],
        [96.0, 98.0, 100.0, 196.0],
        [73.0, 66.0, 70.0, 142.0],
    ],
    dtype=np.float32,
)

# slice data
x = data[:, :-1]
y = data[:, [-1]]

w = tf.Variable(tf.random.normal([3, 1]))
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.000001

# hypothesis, prediction function
def predict(x):
    return tf.matmul(x, w) + b


# 실행 횟수
n_epochs = 2000
for i in range(n_epochs + 1):
    # 비용함수의 gradient를 기록한다.
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(x) - y)))

    # w, b를 각각 할당한다.
    w_grad, b_grad = tape.gradient(cost, [w, b])

    # 파라미터를 업데이트 한다.
    w.assign_sub(learning_rate * w_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))
