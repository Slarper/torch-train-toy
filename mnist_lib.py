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
def print_image(image, filename):
    plt.imshow(np.array(image), cmap='gray')
    plt.savefig(filename)

def print_images(images):
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
    
import struct
import numpy as np
from torch.utils.data import Dataset

class MNISTTagDataset(Dataset):
    def __init__(self, label_file_path):
        
        # Load labels
        with open(label_file_path, 'rb') as lblpath:
            magic, num = struct.unpack(">II", lblpath.read(8))
            self.labels = np.frombuffer(lblpath.read(), dtype=np.uint8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label = self.labels[idx]
        return label
