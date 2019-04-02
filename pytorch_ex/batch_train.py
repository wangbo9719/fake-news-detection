import torch
import torch.utils.data as Data

BATCH_SIZE = 5   #每一批次的数目

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,   #是否打乱样本的顺序
    num_workers=2   #几个线程来工作
)

def show_batch():
    for epoch in range(3): #所有样本训练3次
        for step, (batch_x, batch_y) in enumerate(loader):
            print('epoch:',epoch,',step:',step,',batch_x:',batch_x,',batch_y:',batch_y)

if __name__ == '__main__':
    show_batch()