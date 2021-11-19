from torch import nn
import torch
import numpy as np


x = torch.from_numpy(np.random.random(size=(4, 10)))
print(x.shape)

model = nn.Sequential(
    nn.Linear(in_features=10, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Softmax()
)

ytrue = torch.randint(8, (4, ))
print(ytrue)

loss_fn = nn.CrossEntropyLoss()

print(model(x).shape)
print(ytrue.shape)
loss = loss_fn(model(x), ytrue)

print(torch.randint(5, (3, )))

loss.backward()

for p in model.parameters():
    print(p, p.grad)

