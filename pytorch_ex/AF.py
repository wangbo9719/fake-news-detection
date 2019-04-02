import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

#fake data
x = torch.linspace(-5,5,200) #x data (tensor), shape=(100,1)
#linspace 将从-5到5的线段分成200分
x = Variable(x)
x_np = x.data.numpy()
#画图的时候plt不能使用tensor的格式，所以要转换为numpy

y_relu = torch.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
#y_softplus = F.softplus.data.numpy()

plt.figure(1,figsize = (8,6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2,1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2,1.2))
plt.legend(loc='best')

plt.show()