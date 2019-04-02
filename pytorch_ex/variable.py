import torch
from torch.autograd import Variable
tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad = True)
#通常为true 通过variable搭建了一个计算图纸，requires_grad是否涉及到反向传播
t_out = torch.mean(tensor*tensor) #x^2
v_out = torch.mean(variable*variable)

v_out.backward()
# v_out = 1/4*sum(var*var)
# d(v_out)/d(var) = 1/4 * 2 * variable = 1/2 * variable
#print(variable.grad)
print(variable)  #是Variable形式
print(variable.data)  #是tensor形式
print(variable.data.numpy())