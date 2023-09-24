import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.linear1 = Linear(196608, 10)
    def forward(self, input):
        output = self.linear1(input)
        return output

testModel = TestModel()


for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1,1,1,-1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = testModel(output)
    print(output.shape)
    print("----------------------------------- ")
