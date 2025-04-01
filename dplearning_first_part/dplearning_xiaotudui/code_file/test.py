# root_dir='D:\code_file\dplearning_xiaotutui\data'
# dataset=torchvision.datasets.ImageNet(root_dir, download=True,split='train',transform=torchvision.transforms.ToTensor())
# 数据集太大了放弃
import torchvision
from torchvision.models import VGG16_Weights
import torch



model = torch.load('vgg16.pth', weights_only=False)  # 关闭安全限制
print(model)
