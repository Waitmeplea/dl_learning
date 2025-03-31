# root_dir='D:\code_file\dplearning_xiaotutui\data'
# dataset=torchvision.datasets.ImageNet(root_dir, download=True,split='train',transform=torchvision.transforms.ToTensor())
# 数据集太大了放弃
import torchvision
from torchvision.models import VGG16_Weights




vgg_par_non=torchvision.models.vgg16()
####
vgg_par_non.load_state_dict(torch.load('vgg16.pth'))