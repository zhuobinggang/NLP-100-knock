import numpy as np
import matplotlib.pyplot as plt

filename = "wheat-seeds.csv"
samples_raw = []
samples = []
W1 = -1 
B1 = -1 
W2 = -1 
B2 = -1 
rate = 0.3
units = 4

def reset_w_b():
    global W1,B1,W2,B2
    W1 = np.random.rand(units, 7)
    B1 = np.random.rand(units)
    W2 = np.random.rand(units)
    B2 = np.random.rand(1)

def get_mins_and_maxs(m, lenth):
    mins = []
    maxs = []
    for i in range(lenth):
        maxs.append(np.max(m[:,i]))
        mins.append(np.min(m[:,i]))
    return (np.array(mins),np.array(maxs))

def normalize_samples():
  global samples
  samples = np.array(samples_raw)
  mins,maxs = get_mins_and_maxs(samples, samples.shape[1] - 1)
  max_minus_min = maxs - mins
  mins = np.append(mins, 0)
  max_minus_min = np.append(max_minus_min, 1)
  samples = (samples - mins) / max_minus_min


def read_data_and_prepare_samples():
  global samples, samples_raw, mins, max_minus_min
  with open(filename) as f:
    for line in f.readlines():
      line = line.strip()
      samples_raw.append(line.split(','))
  samples_raw = np.array(samples_raw)
  samples_raw = samples_raw.astype(np.float)
  # normalize input
  normalize_samples()

  # for every col, k
  np.random.shuffle(samples)
  reset_w_b()

read_data_and_prepare_samples()

def relu(x):
    return x if x > 0 else (0.1 * x)

relu = np.vectorize(relu)

def relu_deriv(x):
    return 1 if x > 0 else 0.1

relu_deriv = np.vectorize(relu_deriv)

def is_correct(sample):
    x = sample[0:-1]
    t = sample[-1]
    # Feedforward
    h = np.dot(W1, x) + B1
    a = relu(h)
    z = np.dot(W2, a) + B2
    return 1 if abs(t - z) < 0.5 else 0

def get_correct_rate():
    correct_times = sum(map(is_correct, samples))
    percent = correct_times / samples.shape[0]
    # print('correct rate: ' + str(percent))
    return percent

def run_on_random():
    sample = samples[np.random.randint(0, samples.shape[0])]
    run(sample)

def run(sample):
    global W1, W2, B1, B2
    x = sample[0:-1]
    t = sample[-1]
    # Feedforward
    h = np.dot(W1, x) + B1
    a = relu(h)
    z = np.dot(W2, a) + B2
    # Loss = 0.5 * np.square(t - z[0])
    # print('Before loss:' + str(Loss))
    # backprop
    dz = z - t
    dw2 = dz * a
    db2 = dz
    da = dz * W2
    dh = da * relu_deriv(h)
    db1 = dh
    dw1 = np.outer(dh, x)
    # update
    W1 = W1 - (rate * dw1)
    W2 = W2 - (rate * dw2)
    B1 = B1 - (rate * db1)
    B2 = B2 - (rate * db2)


def run_1000_and_draw():
    global samples
    x = []
    y = []
    for epoch in range(2):
        np.random.shuffle(samples)
        y.append(get_correct_rate() * 100)
        x.append(epoch * samples.shape[0])
        for j in range(samples.shape[0]):
            run(samples[j])
    plt.plot(x, y)
    plt.show()


