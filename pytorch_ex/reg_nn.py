import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


#unsqueeze 把一维数据处理成二维数据
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1) #x data (tensor),shape=(100,1)
y = x.pow(2) + 0.2*torch.rand(x.size())  #y 加上一些噪声的影响

x,y = Variable(x), Variable(y)     #投入网络时一定要转成Variable
#plt.scatter(x.data.numpy(), y.data.numpy())  #画散点图
#plt.show()

'''''''''''''''''''''''''''''''''''''''
网络定义
'''
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net,self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)


    def forward(self,x): #搭建的过程
        x = torch.relu(self.hidden(x))
        x = self.predict(x)  #预测的时候不用激励函数，因为我们不希望预测的结果是被阶段的结果
        return x

net = Net(1, 10, 1) #输入，隐层，输出
print(net)

'''''''''''''''''''''''''''''''''''''''
优化；损失函数
'''
optimizer = torch.optim.SGD(net.parameters(),lr=0.5) #learning rate=0.5，把net的参数送进优化器
#MESLoss 均方差 回归
loss_func = torch.nn.MSELoss()

plt.ion()

'''''''''''''''''''''''''''''''''''''''
训练过程
'''
for t in range(100): #训练的步数
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad() #神经网络参数的梯度先都降为0,因为是循环的，上次的梯度会保存，清空以开始下一次
    loss.backward() #反向传播，计算梯度
    optimizer.step() #用optimizer优化参数

    if t % 5 == 0:
        #plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f'%loss.data,fontdict={'size':12,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()