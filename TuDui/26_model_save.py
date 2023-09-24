import torch
import torchvision

vgg16 = torchvision.models.vgg16()

# 保存方式 1 模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")
# 读取方式
model1 = torch.load("vgg16_method1.pth")
#print(model1)

# 保存方式 2 模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
# 读取方式
vgg16_new = torchvision.models.vgg16()
vgg16_new.load_state_dict(torch.load("vgg16_method2.pth"))
# vgg16_new = torch.load("vgg16_method2.pth")
print(vgg16_new)