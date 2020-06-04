from torch.utils.data import  Dataset, DataLoader
import torch
import numpy as np
import torch.nn as nn

def cate_to_number(cate):
    the_map = {'e':0, 'b':1, 't':2, 'm':3}
    return the_map[cate]

# Prepare dataset

class ReviewDataset(Dataset):
    def __init__(self, filename):
        xy = np.loadtxt(filename, delimiter=',', dtype=str)
        ys = np.array(list(map(cate_to_number, xy[:,0]))).astype(np.long)
        xs = np.array(xy[:,1:]).astype(np.float32)
        xs = xs / 10000 # Normalize
        
        self.y_data = torch.from_numpy(ys)
        self.x_data = torch.from_numpy(xs)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

train_dataset = ReviewDataset('valid.feature.txt')
train_loader = DataLoader(dataset=train_dataset, batch_size=4, num_workers=2)

# Create Network
class Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Network, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = Network(20, 50, 4)

# features, label = dataset[0]

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

softmax = nn.Softmax(dim = 1)

# Training loop
for epoch in range(60):
    for _, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    xs, ys = train_dataset[0:]
    o = softmax(model(xs))
    max_indices = o.max(axis=1).indices
    wrong_count = np.count_nonzero((ys - max_indices).numpy())
    acc = 100 - (wrong_count / train_dataset.n_samples) * 100
    print(f'epoch: {epoch + 1}, acc: {acc}')



