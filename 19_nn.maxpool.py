import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))  # -1是自动计算得来的，即5*5/5/5/1=1
print(input)


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)   # ceil mode这块参照onenote笔记

    def forward(self, input):
        output = self.maxpool1(input)
        return output


testModel = TestModel()
#output = testModel(input)
#print(output)
step = 0
writer = SummaryWriter("logs/19_maxpool")
for data in dataloader:
    print("当前:"+str(step+1)+"/"+str(len(dataloader)))
    imgs, targets = data
    writer.add_images("input2", imgs, step)
    output = testModel(imgs)
    writer.add_images("output2", output, step)
    step = step + 1

writer.close()