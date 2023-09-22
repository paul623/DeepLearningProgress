import torch
import torchvision
from PIL import Image
from model import MyModel
print("人工智障读取数据中")
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
labels_china = ["飞机", "手机", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
img_path = "imgs/paul.png"
image = Image.open(img_path)
image = image.convert('RGB')    # png是四通道，除了RGB三通道之外还有一个透明通道，这一步可以适应png、jpg各种格式的图片
print("...努力判断中....")
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()])
image = transform(image)

model = torch.load("models/MyModel_9.pth", map_location='cpu')
print("...v9模型加载成功....")
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
print("...计算最终结果....")
with torch.no_grad():
    output = model(image)
print("我知道了，这张图片是："+labels_china[output.argmax(1)]+"!!!")

