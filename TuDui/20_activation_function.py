import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid  # <0 均设置为0
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.relu1 = ReLU()  # inplace指的是原地修改，如果设置为true就不需要返回值了。建议默认使用False
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.relu1(input)
        output = self.sigmoid1(input)
        return output


testModel = TestModel()
step = 0
writer = SummaryWriter("logs/20_activation_function")
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = testModel(imgs)
    writer.add_images("output", output, step)
    step = step + 1
    print("进度：" + str(step / len(dataloader)))

writer.close()
