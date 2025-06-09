import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, sampler
import os
# 分布式训练所需的库
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 8)

    def forward(self, x):
        return self.fc2(self.relu1(self.fc1(x)))


x = torch.randn(200, 784)
y = torch.randint(0, 8, size=(200,))
train_data = TensorDataset(x, y)

# 指定后端 backend='nccl' 则是使用gpu gloo则是使用cpu
# 初始化进程组
dist.init_process_group(backend='gloo')
# 赋予该进程一个唯一的本地标识符 (local rank) 运行起来才有意义
local_rank = int(os.environ['LOCAL_RANK'])
print(local_rank)
torch.cuda.set_device(local_rank)
verbose = dist.get_rank() == 0  # 当cuda为0的时候verbose为True

# 模型
model = net().cuda()
# r如果加载参数一个gpu加载就行了
model = DistributedDataParallel(model, device_ids=[local_rank])

train_sampler = DistributedSampler(train_data, shuffle=True)
train_loader = DataLoader(train_data, batch_size=100, num_workers=4, pin_memory=True, shuffle=False,
                          sampler=train_sampler)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss().to(local_rank)

for epoch in range(100):
    model.train()
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(local_rank)
        target = target.to(local_rank)
        optimizer.zero_grad()
        y_hat = model(data)
        l = loss(y_hat, target)
        l.backward()
        optimizer.step()

    if verbose:
        print(local_rank, epoch, l.item())

dist.destroy_process_group()