import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

x, y = Variable(x), Variable(y)

#net
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predicate = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.predicate(x)
        return x

net = Net(2, 10, 2)  #输入两个特征，点和横纵坐标；两个类型




optimizer = torch.optim.SGD(net.parameters(),lr = 0.005)
loss_func = torch.nn.CrossEntropyLoss()
plt.ion()
for t in range(100):
    out = net(x)  #用softmax转换为概率
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        predication = torch.max(F.softmax(out, dim=1),1)[1]
        pred_y = predication.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
