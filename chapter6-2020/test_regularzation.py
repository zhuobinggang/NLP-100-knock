import torch 
import torch.nn as nn

# y = 10 * x + 2


x = torch.tensor([2.])
y = torch.tensor([22.])

def prepare_model():
    model = nn.Linear(1,1)
    with torch.no_grad():
        next(model.parameters())[0][0] = 9.
        model.bias[0] = 2.
    return model

def loss_normal(model):
    loss = 0.5 *  ((y - model(x)) ** 2)
    return loss

def loss_with_regularization(model):
    rate = 0.1
    ps = next(model.parameters())
    loss = 0.5 *  ((y - model(x)) ** 2)
    loss += rate * torch.sum(torch.abs(ps))
    return loss

def loss_only_weight_punish(model):
    rate = 0.1
    ps = next(model.parameters())
    loss = rate * torch.sum(torch.abs(ps))
    return loss


