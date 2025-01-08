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




from safetensors.torch import save_file, load_file

# 加载 .safetensors 文件中的模型参数
loaded_state_dict = load_file("./models/mnist_ae.safetensors")
model.load_state_dict(loaded_state_dict)

# 设置模型为评估模式
model.eval()

# 测试集路径
test_images_path = 'mnist/train-images.idx3-ubyte'
# test_labels_path = 'mnist/t10k-labels.idx1-ubyte'

# 加载测试数据集
test_dataset = MNISTDataset(test_images_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 测试模型性能
with torch.no_grad():
    correct = 0
    total = 0
    # for images in test_loader:
    #     images = images.to(device)
    #     # labels = labels.to(device)
    #     outputs = model(images)
    #     out_images = outputs.detach().cpu().numpy()
    #     # show image
    #     for i in range(64):

    first_batch = next(iter(test_loader))
    inputs = first_batch
    inputs = inputs.to(device)
    outputs = model(inputs)
    # convert outputs into images and print_images them
    out_images = outputs.detach().cpu()
    input_images = inputs.detach().cpu()
    print('Input image shape:', input_images.shape)
    # get shape
    print('Output image shape:', out_images.shape)

    for i in range(64):
        out_ = out_images[i] * 255.0  # scale to [0, 255]
        input_image = input_images[i] * 255.0  # scale to [0, 255]


        input_image = input_image.numpy().reshape((28, 28))    # convert to numpy array
        out_ = out_.numpy().reshape((28, 28))     # convert to numpy array

        # concatenate images horizontally
        image_ = np.hstack((input_image, out_))  # concatenate horizontally



        # save images
        print('Saving concatenated image for input {}...'.format(i))

        print_image(image_, './mnist_output/concatenated_image_verify_{}.png'.format(i))  # save image







        
        # # image = image.view(28, 28)  # reshape to (28, 28)
        # print('Output image {}:'.format(i), out_.shape)

        # print_image(out_, './mnist_output/out_image_{}.png'.format(i))
        # print_image(input_image, './mnist_output/input_image_{}.png'.format(i))














