import torchvision.datasets
from torch.utils.data import DataLoader

test_data = torchvision.datasets.MNIST(root="../../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                       download=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
