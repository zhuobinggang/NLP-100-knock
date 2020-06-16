import prepare_dict as pd
import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.GRU(2000, 50)
        self.output_w = nn.Linear(50, 4)

    def forward(self, x): # x: (seq_len, batch_size, input_size)
        _, o = self.rnn(x)  # Many to one, only get the last one, (layer_size == 1, batch_size, hidden_size)
        o = self.output_w(o[0]) # (batch_size, output_size)
        return o

def preprocess_samples(raw_samples):
    res = []
    for cate, words in raw_samples:
        res.append((cate, torch.tensor(words, dtype=torch.float32)))
    return res

samples = None
loss = nn.CrossEntropyLoss()
model = Net()
optim = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def init_samples():
    global samples
    samples = preprocess_samples(pd.get_samples())

def get_acc():
    # output acc
    with torch.no_grad():
        hit_cnt = 0
        for label, inp in samples: 
            o = model(inp)[0]
            if torch.max(o)  == o[label]:
                hit_cnt += 1
        acc = {hit_cnt / len(samples) * 100}
        return acc

def run_by_epochs(epochs):
    print(f'Start acc: {get_acc()}')
    for epoch in range(epochs):
        for label, inp in samples:  # label: int, inp: (seq_len, batch_size, input_size)
            optim.zero_grad()
            o = model(inp)
            l = loss(o, torch.tensor([label]))
            l.backward()
            optim.step()
        print(f'epoch{epoch+1} acc: {get_acc()}')


