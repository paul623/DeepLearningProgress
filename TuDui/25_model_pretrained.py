import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(weights= torchvision.models.VGG16_Weights.DEFAULT)

train_data = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# x下面的几种常见的修改方式，效果都一样

# vgg16.add_module('add_linear', nn.Linear(1000, 10))
# print(vgg16)
#
# vgg16.classifier.add_module('add_linear', nn.Linear(1000, 10))
# print(vgg16)

vgg16.classifier[6] = nn.Linear(4096, 10)