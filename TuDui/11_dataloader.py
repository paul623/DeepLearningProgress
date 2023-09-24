import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中第一章图片以及target
img, target = test_data[0]
print(img)
print(target)

writer = SummaryWriter("logs/11_dataloader")
# 测试数据集使用dataLoder后的结果
step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("Epoch:{}".format(), imgs, step)
    step = step+1
writer.close()