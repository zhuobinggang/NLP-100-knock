import matplotlib.pyplot as plt
import numpy as np

def plot_acc_vs_regul_param(matrix, xlabels):
    # matrix = np.array([[92.3, 87.3, 77.3],[92.3, 87.3, 77.3], [92.3, 87.3, 77.3], [92.3, 87.3, 77.3], [92.3, 87.3, 77.3]])
    # matrix = np.array([[92.3, 87.3, 77.3, 80, 80],[92.3, 87.3, 77.3, 80, 80], [92.3, 87.3, 77.3, 80, 80], [92.3, 87.3, 77.3, 80, 80], [92.3, 87.3, 77.3, 80, 80]])
    # xlabels = np.array(['10', '1', '0.1', '0.01', '0.001'])
    xs = np.array(range(len(xlabels)))
    plt.xticks(xs, xlabels)
    width = 1/5
    plt.bar(xs - (width * 1), matrix[:,0], width = width, label="dd")
    plt.bar(xs, matrix[:,1], width = width)
    plt.bar(xs + (width * 1), matrix[:,2], width = width)
    plt.show()


