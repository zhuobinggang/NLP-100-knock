import torch
import torch.nn as nn


learning_rate = 0.01
n_iters = 10

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[4]], dtype=torch.float32)

model = nn.Linear(1, 1)


loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(n_iters):
    y_pred = model(X)
    l = loss(Y, y_pred)
    l.backward()

    optimizer.step()
    optimizer.zero_grad()


    if epoch % 1== 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

