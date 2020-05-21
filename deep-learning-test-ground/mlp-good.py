import numpy as np
import random

def relu(v):
    return np.maximum(v, 0)

def get_h(x, w1, b1):
    return np.dot(x, w1.T) + b1

def once(x, label, w1, b1, w2, b2):
    h = get_h(x, w1, b1) 
    a = relu(h)
    z = get_h(a, w2, b2)
    y = relu(z)
    t = label
    Loss = 1/2 * np.square(t - y[0])
    print('Loss: ' + str(Loss))
    # Backprop
    dy = y[0] - t
    dz = dy * (1 if z[0] > 0 else 0) 
    db2 = dz 
    dw2 = dz * a

    da = dz * w2 ################

    dh = np.array([hi * da[index] for index,hi in enumerate(map(lambda it: 1 if it >= 0 else 0, h))])
    dw1 = np.outer(dh, x)
    db1 = dh
    # training
    rate = 0.1
    new_w1 = w1 - rate * dw1
    new_b1 = b1 - rate * db1
    new_w2 = w2 - rate * dw2
    new_b2 = b2 - rate * db2
    return (new_w1, new_b1, new_w2, new_b2)

# Initial
samples = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
inputs = samples[:, 0:2]
labels = samples[:, 2]
W1 = np.random.rand(2,2)
B1 = np.random.rand(2)
W2 = np.random.rand(2)
B2 = np.random.rand(1)

def run():
    global W1,W2,B1,B2
    index = random.randint(0,3)
    W1,B1,W2,B2 = once(inputs[index], labels[index], W1, B1, W2, B2)
    output_loss(inputs[index], labels[index])

def output_loss(x, t):
    h = get_h(x, W1, B1) 
    a = relu(h)
    z = get_h(a, W2, B2)
    y = relu(z)
    Loss = 1/2 * np.square(t - y)
    print('After Loss: ' + str(Loss), t, y, '\n')

def compare(x):
    h = get_h(np.array(x), W1, B1) 
    a = relu(h)
    z = get_h(a, W2, B2)
    y = relu(z)
    print(y)

for i in range(10000):
    run()

compare([0,0])
compare([0,1])
compare([1,0])
compare([1,1])
