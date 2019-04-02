import torch
import numpy as np
#生成一个五行四列的二维矩阵
print(torch.Tensor(5, 4))
# 返回的数组大小是5x4的矩阵，初始化是0~1的均匀分布
x = torch.rand(5, 4)
print(torch.randn(5, 4))
print(x)
#查看x的形状
print(x.size())
# numpy 类似的返回5x4大小的矩阵
print(np.ones((5, 4)))
#  类似的返回5x4大小的张量
print(torch.ones(5,4))
#返回5x4大小的张量 对角线上全1，其他全0
print(torch.eye(5,4))
print(torch.arange(1,5,1))
print(torch.linspace(1,5,2))
#服从正太分布
#print(torch.normal(-1,2))
print(torch.randperm(2))
#numpy转换成Tensor
a = torch.ones(5)
b=a.numpy()
print(b)
#Tensor转换成numpy
a= np.ones(5)
b=torch.from_numpy(a)
print(b)
#True支持GPU，False不支持
print(torch.cuda.is_available())