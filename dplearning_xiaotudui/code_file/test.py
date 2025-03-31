from sqlalchemy.util import OrderedDict
from torch import nn, optim
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
root=r'D:\bigdata\dl_learning\dplearning_xiaotudui\data'
transform = transforms.Compose([transforms.ToTensor()])
data_set=torchvision.datasets.CIFAR10(root,train=False,transform=transform,download=False)
data=DataLoader(data_set,batch_size=10,shuffle=True)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.conv1=nn.Conv2d(3,32,5,padding=2)
        # self.pool1 = nn.MaxPool2d(2,2)
        # self.conv2=nn.Conv2d(32,32,5,padding=2)
        # self.pool2 = nn.MaxPool2d(2,2)
        # self.conv3=nn.Conv2d(32,64,5,padding=2)
        # self.pool3 = nn.MaxPool2d(2,2)
        # self.flatten = nn.Flatten()
        # self.linear1=nn.Linear(1024,64)
        # ###一共10 类别 因此要64*10
        # self.linear2=nn.Linear(64,10)

        self.sequential = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,32,5,padding=2),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5,padding=2),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )
        self.softmax1=nn.Softmax(dim=1)
    def forward(self,x):
        x=self.sequential(x)
        # x=self.softmax1(x)
        return x

net=Net()
loss=nn.CrossEntropyLoss()

optm=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
for i in data:

    imgs,targets=i
    result=net(imgs)
    loss_result=loss(result,targets)
    optm.zero_grad()
    loss_result.backward()
    optm.step()