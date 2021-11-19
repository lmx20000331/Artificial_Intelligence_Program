import numpy as np
import torch

x = np.random.random(size=(100, 8))

linear = torch.nn.Linear(in_features=8, out_features=1)
sigmoid = torch.nn.Sigmoid()
linear2 = torch.nn.Linear(in_features=1, out_features=1)

model = torch.nn.Sequential(linear, sigmoid, linear2).double()

train_x = torch.from_numpy(x)

print(model(train_x).shape)

yture = torch.from_numpy(np.random.uniform(0, 5, size=(100, 1)))

# print(x)
print(yture.shape)

loss_fn = torch.nn.MSELoss()

optimer = torch.optim.SGD(model.parameters(), lr=1e-5)

for e in range(100):
    for b in range(100 // 1): # stochastic gradient descent
    # for b in range(100 // 10): # mini-batch gradient descent
    # for b in range(100 // 100): # batch gradient descent
        batch_index = np.random.choice(range(len(train_x)), size=20)

        yhat = model(train_x[batch_index])
        loss = loss_fn(yhat, yture[batch_index])
        loss.backward()
        print(loss)
        optimer.step()
