from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

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


def init_model_loss_optim():
    global model, criterion, optimizer
    model = Network(2000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


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




