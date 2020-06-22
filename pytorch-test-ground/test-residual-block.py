import torch
import torch.nn as nn

sigmoid = nn.Sigmoid()
linear = nn.Linear(1,1)

def forward_tradictional(x):
    # y = F(x) + x
    o = sigmoid(linear(x))
    return o

def forward_residual(x):
    o = sigmoid(linear(x))
    return o + x

