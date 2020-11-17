import numpy as np

x = np.array([1, 2, 3])
y = np.array([1, 2, 3])


def cost_func(w, x, y):
    c = 0
    for i in range(len(x)):
        c += (w * x[i] - y[i]) ** 2

    return c / len(x)


for feed_w in np.linspace(-3, 5, num=15):
    curr_cost = cost_func(feed_w, x, y)
    print("{:6.3f} | {:10.5f}".format(feed_w, curr_cost))

