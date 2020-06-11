from gensim.models.keyedvectors import KeyedVectors
from gensim.utils import simple_preprocess
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

model = None

def init_model():
    global model
    print('Loading trained model...')
    model = KeyedVectors.load_word2vec_format('../chapter7-2020/GoogleNews-vectors-negative300.bin', binary=True)
    print('Model is ready.')

def simple_preprocess_data(in_file = 'train.txt', out_file='train.processed.txt'):
    # read and simple preprocess data by gensim
    lines = []
    with open(in_file) as f:
        for line in f:
            cate, title = line.strip().split('\t')
            words = simple_preprocess(title)
            title_new = ','.join(words)
            lines.append(f'{cate} {title_new}')
    content = '\n'.join(lines)
    with open(out_file, 'w') as f:
        f.write(content)

def preprocess_feature_data(in_file = 'train.processed.txt', out_file = 'train.feature.txt'):
    if model is None:
        init_model()
    lines = []
    with open(in_file) as f:
        for line in f:
            cate, title = line.strip().split(' ')
            words = title.split(',')
            # Turn word to vector and accumulate them
            acc_vec = np.zeros(300)
            word_cnt = 0
            for word in words:
                try:
                    vec = model.get_vector(word)
                    acc_vec += vec
                    word_cnt += 1
                except KeyError as e:
                    pass
            acc_vec = acc_vec / word_cnt
            title_vec = ','.join(acc_vec.astype(str))
            lines.append(f'{cate},{title_vec}')
    content = '\n'.join(lines)
    with open(out_file,'w') as f:
        f.write(content)

def prepare_feature_files():
    simple_preprocess_data() # default is train 
    simple_preprocess_data('test.txt', 'test.processed.txt')
    simple_preprocess_data('valid.txt', 'valid.processed.txt')
    preprocess_feature_data() # default is train 
    preprocess_feature_data('test.processed.txt', 'test.feature.txt')
    preprocess_feature_data('valid.processed.txt', 'valid.feature.txt')

# Construct Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(300, 4)

    def forward(self, x):
        out = self.l1(x)
        return out

def cate_to_number(cate):
    the_map = {'e':0, 'b':1, 't':2, 'm':3}
    return the_map[cate]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename = 'train.feature.txt'):
        super(Dataset).__init__()
        labels = []
        features = []
        with open(filename) as f:
            for line in f:
                cate_features = line.split(',')
                labels.append(cate_to_number(cate_features[0]))
                features.append(cate_features[1:])
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.features = torch.from_numpy(np.array(features, dtype='float32'))
        self.length = len(self.labels)
    def __getitem__(self, index):
        return self.labels[index], self.features[index]
    def __len__(self):
        return self.length

train_dataset = None
test_dataset = None
data_loader = None
criterion = None
optimizer = None
model = None
softmax = nn.Softmax(dim = 1)


def init_dnn_components():
    global train_dataset, test_dataset, data_loader, criterion, optimizer, model
    model = Net()
    train_dataset = Dataset('train.feature.txt')
    # Prepare the test.feature.txt
    test_dataset = Dataset('test.feature.txt')
    data_loader = DataLoader(train_dataset, batch_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def get_accuracy(dataset):
    ys, xs = dataset[0:]
    o = softmax(model(xs))
    max_indices = o.max(axis=1).indices
    wrong_count = np.count_nonzero((ys - max_indices).numpy())
    acc = 100 - (wrong_count / dataset.length) * 100
    return acc


def get_loss(dataset):
    with torch.no_grad():
        ys, xs = dataset[0:]
        o = softmax(model(xs))
        loss = criterion(o, ys)
    return loss / dataset.length

def plt_loss_and_accuracy(axis_x, loss1, acc1, loss2, acc2):
    plt.subplot(1, 2, 1)
    plt.plot(axis_x, loss1, 'b', label='Train loss')
    plt.plot(axis_x, loss2, 'g', label='Test loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(axis_x, acc1, 'b', label='Train accuracy')
    plt.plot(axis_x, acc2, 'g', label='Test accuracy')
    plt.legend()
    plt.show()


def run_by_epochs(epochs):
    # Record loss by train_dataset & test_dataset
    train_loss = []
    test_loss = []
    # Record accuracy by train_dataset & test_dataset
    train_accuracy = []
    test_accuracy = []
    for epoch in range(epochs):
        for target, x in data_loader:
            out = model(x)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_accuracy.append(get_accuracy(train_dataset))
        test_accuracy.append(get_accuracy(test_dataset))
        train_loss.append(get_loss(train_dataset))
        test_loss.append(get_loss(test_dataset))
    # Draw them by plt
    return list(range(1, epochs + 1)), train_loss, train_accuracy, test_loss, test_accuracy




