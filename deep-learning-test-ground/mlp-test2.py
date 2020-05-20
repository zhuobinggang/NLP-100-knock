from random import seed
from random import random
import numpy as np

def raw_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

sigmoid = np.vectorize(raw_sigmoid)

samples = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]])
inputs = samples[:, 0:2]
labels = samples[:, 2]

# print(samples[:,2])

seed(1)

def raw_minus_zero_point5(x):
    return x - 0.5
minus_zero_point5 = np.vectorize(raw_minus_zero_point5)

W1 = np.random.rand(2, 2)
W2 = np.random.rand(2, 2)
# print(W1)
# W2 = np.random.rand(2, 2)

def softmax(xs):
    exps = [np.exp(x) for x in xs]
    sum_exps = sum(exps)
    return [exp/sum_exps for exp in exps]
    

# W1 = np.array([[1, 2],[3, 4]])
# W2 = np.array([[1, 2],[3, 4]])
# dd2 = np.dot(dd, W2.T)

output1 = sigmoid(np.dot(inputs, W1.T))
output2 = np.dot(output1, W2.T)

result = [softmax(output_per_sample) for output_per_sample in output2]

print(result)

