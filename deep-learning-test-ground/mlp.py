import numpy as np
import random
import matplotlib.pyplot as plt


rate = 0.5

def relu(v):
    return np.maximum(v, 0)

def get_h(x, w1, b1):
    return np.dot(x, w1.T) + b1

def raw_deriv_relu(x):
    return 1 if x > 0 else 0

deriv_relu = np.vectorize(raw_deriv_relu)

def get_cost(t, y):
    return (1 / (2 * len(t))) * np.sum(np.square(t - y))


def check(x, labels, w1, b1, w2, b2):
    h = get_h(x, w1, b1) 
    a = relu(h)
    z = get_h(a, w2, b2)
    y = relu(z)
    t = labels
    # Loss = 1/2 * np.square(t - y)
    Cost = get_cost(t, y)
    print('Cost: ' + str(Cost))
    # print('Loss: ' + str(Loss))
    # Backprop
    dy = (y - t) / len(t)
    dz = dy * deriv_relu(z) 
    db2 = dz.sum(axis=0)
    dw2 = np.dot(dz.T, a)

    # Check dy correct
    h = 1e-5
    # dy
    before = np.array(y)
    before[0] = before[0] - h
    after = np.array(y)
    after[0] = after[0] + h
    dy_raw = (get_cost(t, after) - get_cost(t, before)) / (2 * h)
    print('dy_raw')
    print(dy_raw)
    print('dy')
    print(dy)
    # dz
    # db2
    # dw2




def once(x, labels, w1, b1, w2, b2):
    h = get_h(x, w1, b1) 
    a = relu(h)
    z = get_h(a, w2, b2)
    y = relu(z)
    t = labels
    # Loss = 1/2 * np.square(t - y)
    # Cost = 1 / (2 * len(t)) * np.sum(np.square(t - y))
    # print('Cost: ' + str(Cost))
    # print('Loss: ' + str(Loss))
    # # Backprop
    dy = (y - t) / len(t)
    dz = dy * deriv_relu(z) 
    db2 = dz.sum(axis=0)
    dw2 = np.dot(dz.T, a)
    # da = np.dot(dz * w2)
    da = np.outer(dz, w2)
    dh = da * deriv_relu(h)
    db1 = dh.sum(axis=0)
    dw1 = np.dot(dh.T, x)

    # training
    new_w1 = w1 - rate * dw1
    new_b1 = b1 - rate * db1
    new_w2 = w2 - rate * dw2
    new_b2 = b2 - rate * db2
    return (new_w1, new_b1, new_w2, new_b2)


# Initial
samples = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
inputs = samples[:, 0:2]
labels = samples[:, 2]

hidden_units = 4
W1 = -1
B1 = -1
W2 = -1
B2 = -1


def reset_w_and_b():
  global W1,W2,B1,B2
  W1 = np.random.rand(hidden_units,2)
  B1 = np.random.rand(hidden_units)
  W2 = np.random.rand(hidden_units)
  B2 = np.random.rand(1)

reset_w_and_b()

# W1 = np.random.rand(2,2)
# B1 = np.random.rand(2)
# W2 = np.random.rand(2)
# B2 = np.random.rand(1)

def run():
    global W1,W2,B1,B2
    W1,B1,W2,B2 = once(inputs, labels, W1, B1, W2, B2)

def run_check():
    check(inputs, labels, W1, B1, W2, B2)

def output_loss(x, t):
    h = get_h(x, W1, B1) 
    a = relu(h)
    z = get_h(a, W2, B2)
    y = relu(z)
    t = labels
    # Loss = 1/2 * np.square(t - y)
    Cost = 1 / (2 * len(t)) * np.sum(np.square(t - y))
    return Cost


def compare2():
  for index,x in enumerate(inputs):
    h = get_h(x, W1, B1) 
    a = relu(h)
    z = get_h(a, W2, B2)
    y = relu(z)
    t = labels[index]
    print('z, t: ' + str(z[0]) + ', ' + str(t))
    Loss = 1/2 * np.square(t - y)
    print('Loss: ' + str(Loss))


def output_dw():
    w1 = W1
    w2 = W2
    b2 = B2
    b1 = B1
    x = inputs
    h = get_h(x, W1, B1) 
    a = relu(h)
    z = get_h(a, W2, B2)
    y = relu(z)
    t = labels
    # Loss = 1/2 * np.square(t - y)
    dy = (y - t) / len(t)
    dz = dy * deriv_relu(z) 
    db2 = dz.sum(axis=0)
    dw2 = np.dot(dz.T, a)
    # da = np.dot(dz * w2)
    da = np.outer(dz, w2)
    dh = da * deriv_relu(h)
    db1 = dh.sum(axis=0)
    dw1 = np.dot(dh.T, x)
    print('Gradient of dw1')
    print(dw1)
    print('Gradient of db1')
    print(db1)
    print('Gradient of dw2')
    print(dw2)
    print('Gradient of db2')
    print(db2)


def run1000_and_draw():
    x = [0]
    y = []
    y.append(output_loss(inputs, labels))
    for i in range(1000):
        run()
        x.append(i+1)
        y.append(output_loss(inputs, labels))
    plt.plot(x, y)
    plt.title('Hidden Units: ' + str(hidden_units) + ' Learning rate: ' + str(rate))
    plt.show()


