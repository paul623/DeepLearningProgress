import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloder = DataLoader(dataset, batch_size=64)


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


testModel = TestModel()

writer = SummaryWriter("logs/18")
step = 0
for data in dataloder:
    imgs, targets = data
    output = testModel(imgs)
    print("--------------------------------------------")
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30]) -> [xxx, 3, 30, 30]

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step = step+1

writer.close()