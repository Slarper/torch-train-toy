import struct
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, filename, transform=None):
        self.transform = transform
        with open(filename, 'rb') as f:
            # 解析文件头信息
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            print(f'Magic number: {magic}, Image count: {num_images}, Rows: {rows}, Columns: {cols}')

            # 将所有图像存储在列表中
            self.images = []
            image_size = rows * cols
            fmt = '>' + str(image_size) + 'B'
            for _ in range(num_images):
                image = struct.unpack(fmt, f.read(image_size))
                # 并转换成 28x28 的矩阵
                image_matrix = np.array(image).reshape((28, 28)).astype(np.uint8)
                self.images.append(image_matrix)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

#  dataloader is way more faster
def read_images(filename):
    with open(filename, 'rb') as f:
        # 解析文件头信息
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        print(f'Magic number: {magic}, Image count: {num_images}, Rows: {rows}, Columns: {cols}')

        # 解析图像数据
        image_size = rows * cols
        fmt = '>' + str(image_size) + 'B'  # 大端字节序，每个像素占一个字节
        for _ in range(num_images):
            image = struct.unpack(fmt, f.read(image_size))
            # 并转换成 28x28 的矩阵
            image_matrix = np.array(image).reshape((28, 28))
            yield image  # 使用生成器逐个返回图像 just a tuple


# 显示图像的函数


def show_images(images):
    for i, image in enumerate(images):
        image_matrix = np.array(image).reshape((28, 28))
        plt.imshow(image_matrix, cmap='gray')
        plt.savefig(f'mnist_output/image_{i}.png')  # 保存图像到文件


from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义自动编码器类
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # 假设输入是28x28像素的图像
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)  # 潜在空间维度为3
        )
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # 输出应该与输入范围相匹配，假设输入被归一化到[0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化模型、损失函数和优化器
model = Autoencoder().to(device)  # 将模型移动到设备上（CPU或GPU）
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.view(-1))])  # 将图像展平成向量

# 加载MNIST数据集
# train_loader = read_images('mnist/train-images.idx3-ubyte')
# 创建数据集实例
train_dataset = MNISTDataset('mnist/train-images.idx3-ubyte', transform=transform)
# 创建 DataLoader 实例
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



print(f"Using device: {device}")


# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for data in train_loader:
        # tensor = torch.tensor(data, dtype=torch.float32) / 255.0  # 归一化像素值到[0, 1]范围

        imgs = data  # 我们不需要标签
        imgs = imgs.view(imgs.size(0), -1)  # 展平图片
        imgs = imgs.to(device)  # 将数据移动到设备上

        
        # 前向传播
        output = model(imgs)
        loss = criterion(output, imgs)  # 自动编码器尝试最小化输入和输出之间的差异
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("训练完成")
