from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

train_dataset = None
train_loader = None
model = None
criterion = None
optimizer = None
softmax = nn.Softmax(dim = 1)

def cate_to_number(cate):
    the_map = {'e':0, 'b':1, 't':2, 'm':3}
    return the_map[cate]

def cate_index_catename(index):
    the_map = {0: 'entertainment', 1: 'finance', 2: 'technology', 3: 'health'}
    return the_map[index]

def words_to_bag(word_index_map, words, dim=2000):
    res = np.zeros(dim)
    for word in words:
        if word in word_index_map:
            res[word_index_map[word]] = 1
    return res


def init_word_index_map():
    with open('high_frequency_words.csv') as f:
        lines = f.read().strip().split('\n')
        word_count_pairs = list(map(lambda line: line.strip().split(','), lines))
        word_count_pairs = np.array(word_count_pairs)
        words = word_count_pairs[:,0]
        res = {}
        for index, word in enumerate(words):
            res[word] = index
        return res




class ReviewDataset(Dataset):
    def __init__(self, filename):
        super(ReviewDataset).__init__()
        word_index_map = init_word_index_map()
        labels = [] # dim: N
        features = [] # dim: (N,2000)
        with open(filename) as f:
            for line in f.readlines():
                cate, title = line.split('\t')
                words = title.strip().split(' ')
                labels.append(cate_to_number(cate))
                features.append(words_to_bag(word_index_map, words))
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.length = self.labels.shape[0]

    def __getitem__(self, key):
        return self.labels[key], self.features[key]

    def  __len__(self):
        return self.length


def init_dataset_and_loader():
    global train_dataset, train_loader
    train_dataset = ReviewDataset('train.formatted.txt')
    train_loader = DataLoader(train_dataset, 4)

# Create Network


class Network(nn.Module):
    def __init__(self, input_size):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_size, 10)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(10, 4)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out


def init_model_loss_optim(decay=0):
    global model, criterion, optimizer
    model = Network(2000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=decay)


def run_by_epochs(epochs = 10):
    # Train loop
    for epoch in range(epochs):
        for _,(labels, features) in enumerate(train_loader):
            o = model(features)
            loss = criterion(o, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ouput acc
        ys, xs = train_dataset[0:]
        o = softmax(model(xs))
        max_indices = o.max(axis=1).indices
        wrong_count = np.count_nonzero((ys - max_indices).numpy())
        acc = 100 - (wrong_count / train_dataset.length) * 100
        print(f'epoch: {epoch + 1}, acc: {acc}')

def load_saved_model():
    global model
    model = Network(2000)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

def cal_accuracy_by_dataset(dataset):
    ys, xs = dataset[0:]
    o = softmax(model(xs))
    max_indices = o.max(axis=1).indices
    wrong_count = np.count_nonzero((ys - max_indices).numpy())
    acc = 100 - (wrong_count / dataset.length) * 100
    return acc

def run_test():
    test_dataset = ReviewDataset('test.formatted.txt')
    acc = cal_accuracy_by_dataset(test_dataset)
    print('++++++++++++')
    print(f'Test Acc: {acc}')

def run_valid():
    valid_dataset = ReviewDataset('valid.formatted.txt')
    acc = cal_accuracy_by_dataset(valid_dataset)
    print('++++++++++++')
    print(f'Valid Acc: {acc}')

def print_out_confusion_matrix(m):
    content = f'{"": >25}{cate_index_catename(0): >20}{cate_index_catename(1): >20}{cate_index_catename(2): >20}{cate_index_catename(3): >20}\n'
    for i in range(len(m)):
        head = f'{cate_index_catename(i)}(expected)'
        tail = ''.join(list(map(lambda x: f'{x: >20}', m[i])))
        content += f'{head: >25}{tail}\n'
    print(content)

# dataset = ReviewDataset('test.formatted.txt')

def get_test_confusion_matrix(dataset):
    labels, xs = dataset[0:]
    predicted = softmax(model(xs)).max(axis=1).indices
    res = np.zeros(16).reshape(4,4)
    for i in range(len(labels)):
        res[predicted[i]][labels[i]] += 1
    return res.astype(np.int)
    
def cal_precision_recall_f1score_of_cates(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        print('Wrong matrix!')
        return
    precision = []
    recall = []
    for i in range(matrix.shape[0]):
        tp = matrix[i][i]
        fp_add_tp = np.sum(matrix[i])
        precision.append(tp / fp_add_tp)
        tp_add_fn = np.sum(matrix[:,i])
        recall.append(tp/tp_add_fn)
    f1score = []
    for i in range(len(precision)):
        p = precision[i]
        r = recall[i]
        f1score.append(2 / (1/p + 1/r))
    return precision, recall, f1score

def cal_micro_avg(matrix):
    precision_avg = 0
    sum_predicted = 0
    sum_fact = 0
    sum_true_predicted = 0
    for i in range(matrix.shape[0]):
        sum_predicted += np.sum(matrix[i])
        sum_fact += np.sum(matrix[:, i])
        sum_true_predicted += matrix[i, i]
    micro_precision = sum_true_predicted / sum_predicted
    micro_recall = sum_true_predicted / sum_fact
    micro_f1 = 2 / (1/micro_precision + 1/micro_recall)
    return micro_precision, micro_recall, micro_f1
   
def cal_macro_avg(matrix):
    p, r, f = cal_precision_recall_f1score_of_cates(matrix)
    return np.sum(p)/4, np.sum(r)/4, np.sum(f)/4

def q56(m):
    # dataset = ReviewDataset('test.formatted.txt')
    # m = get_test_confusion_matrix(dataset)
    p, r, f = cal_precision_recall_f1score_of_cates(matrix)
    content = f"{'': >20}{'precision': >20}{'recall': >20}{'f1-score': >20}\n"
    for i in range(len(p)):
        content += f"{cate_index_catename(i): >20}{p[i]: >20}{r[i]: >20}{f[i]: >20}\n"
    print(content)
        

def high_frequency_words():
    with open('high_frequency_words.csv') as f:
        lines = f.read().strip().split('\n')
        word_count_pairs = list(map(lambda line: line.strip().split(','), lines))
        return word_count_pairs


def get_most_heavy_weights(words = None):
    if model is None:
        return
    if words is None:
        words = high_frequency_words()
    w = next(model.layer1.parameters()).clone().detach()
    feature_weight_sums = torch.sum(w, axis=0)
    weight_index_pairs = [(x,i) for i,x in enumerate(feature_weight_sums)]
    weight_index_pairs = list(reversed(sorted(weight_index_pairs, key = lambda p: p[0])))
    indices = list(map(lambda x: x[1] ,weight_index_pairs))
    return [words[i] for i in indices[0: 20]]
    

def get_acc_matrix_on_different_regula_param(decays):
    # init datasets
    init_dataset_and_loader()
    test_dataset = ReviewDataset('test.formatted.txt')
    valid_dataset = ReviewDataset('valid.formatted.txt')

    if decays is None:
        decays = [0.05, 0.01, 0.001, 0.0001] 

    # for each decay rate in [0.05, 0.01, 0.001, 0.0001]
    res = []
    for decay in decays:
        init_model_loss_optim(decay)
        run_by_epochs(5)
        res.append([cal_accuracy_by_dataset(train_dataset),
            cal_accuracy_by_dataset(test_dataset),
            cal_accuracy_by_dataset(valid_dataset)])
    return np.array(res), decays


def plot_acc_vs_regul_param(matrix, xlabels):
    # matrix = np.array([[92.3, 87.3, 77.3],[92.3, 87.3, 77.3], [92.3, 87.3, 77.3], [92.3, 87.3, 77.3], [92.3, 87.3, 77.3]])
    # matrix = np.array([[92.3, 87.3, 77.3, 80, 80],[92.3, 87.3, 77.3, 80, 80], [92.3, 87.3, 77.3, 80, 80], [92.3, 87.3, 77.3, 80, 80], [92.3, 87.3, 77.3, 80, 80]])
    # xlabels = np.array(['10', '1', '0.1', '0.01', '0.001'])
    xs = np.array(range(len(xlabels)))
    plt.xticks(xs, xlabels)
    width = 1/5
    plt.bar(xs - (width * 1), matrix[:,0], width = width, label=xlabels[0])
    plt.bar(xs, matrix[:,1], width = width, label=xlabels[1])
    plt.bar(xs + (width * 1), matrix[:,2], width = width, label=xlabels[2])
    plt.legend()
    plt.show()

