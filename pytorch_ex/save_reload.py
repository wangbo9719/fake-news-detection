import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn

x = torch.unsqueeze(torch.linspace(-1,1,100), dim = 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

def save():
    net1 = nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr = 0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1, figsize = (10, 3))
    plt.subplot(131)
    plt.title('net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 5)

    torch.save(net1, 'net.pkl')
    torch.save(net1.state_dict(), 'net_paras.pkl')

def restore():
    net2 = torch.load('net.pkl')
    prediction = net2(x)

    plt.subplot(132)
    plt.title('net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

def restore_paras():
    net3 = nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load('net_paras.pkl'))
    prediction = net3(x)

    plt.subplot(133)
    plt.title('net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


save()
restore()
restore_paras()
plt.show()