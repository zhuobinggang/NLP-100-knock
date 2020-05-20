import matplotlib.pyplot as plt
import numpy as np

def plot_data(data, labels):
    """
    argument:
    data: np.array containing the input value
    labels: 1d numpy array containing the expected label
    """
    positives = data[labels == 1, :]
    negatives = data[labels == 0, :]
    plt.scatter(positives[:, 0], positives[:, 1], 
                       color='red', marker='+', s=200)
    plt.scatter(negatives[:, 0], negatives[:, 1], 
                       color='blue', marker='_', s=200)
    plt.show()

positives = np.array([[1, 0], [0, 1]])
negatives = np.array([[0, 0], [1, 1]])

data = np.concatenate([positives, negatives])
labels = np.array([1, 1, 0, 0])
plot_data(data, labels)
