import torch
import numpy as np
'''
np_data = np.arange(6).reshape(2,3) #两行三列
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print(np_data,"\n",torch_data,"\n",tensor2array)
'''

#abs
data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)

print(
    '\nnumpy:',np.matmul(data,data),
    '\ntorch:',torch.mm(tensor, tensor)
)