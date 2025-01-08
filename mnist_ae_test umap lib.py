import struct
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader



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

from mnist_lib import Autoencoder, MNISTDataset


    
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
# train_dataset = MNISTDataset('mnist/train-images.idx3-ubyte', transform=transform)
# 创建 DataLoader 实例
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



print(f"Using device: {device}")




from safetensors.torch import save_file, load_file

# 加载 .safetensors 文件中的模型参数
loaded_state_dict = load_file("./models/mnist_ae.safetensors")
model.load_state_dict(loaded_state_dict)

# 设置模型为评估模式
model.eval()

# 测试集路径
test_images_path = 'mnist/t10k-images.idx3-ubyte'
test_labels_path = 'mnist/t10k-labels.idx1-ubyte'

# 加载测试数据集
test_dataset = MNISTDataset(test_images_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


# Assuming you want to visualize activations from fc12
def get_activations(model, x_input):
    with torch.no_grad():
        model.eval()
        # Forward pass until the layer you want to visualize
        activations = model.encoder(x_input)
    return activations

import umap
from mnist_lib import MNISTTagDataset
from matplotlib.colors import ListedColormap
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
    # outputs = model(inputs)
    # # convert outputs into images and print_images them
    # out_images = outputs.detach().cpu()
    # input_images = inputs.detach().cpu()
    # print('Input image shape:', input_images.shape)
    # # get shape
    # print('Output image shape:', out_images.shape)


    test_label_dataset = MNISTTagDataset(label_file_path=test_labels_path)
    test_label_loader = DataLoader(test_label_dataset, batch_size=len(test_label_dataset), shuffle=False)

    # Get labels for the first batch of test images
    labels = next(iter(test_label_loader))
    
    activations = get_activations(model, inputs).detach().cpu().numpy()

    # Apply UMAP to reduce dimensions to 2D for visualization
    reducer = umap.UMAP(n_components=2, random_state=42)
    # reducer = umap.UMAP(min_dist=0.5)
    embedding = reducer.fit_transform(activations)
    # print embedding shape
    print("Embedding shape:", embedding.shape)


    # input_cpu = inputs.detach().cpu().numpy()


    # Define a list of 10 distinct colors for categories 0 through 9
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'   # blue-teal
    ]

    # Create a ListedColormap from the list of colors
    cmap = ListedColormap(colors)

    figsize = (10, 7)


    # Plot the embeddings
    plt.figure(figsize=figsize)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=input_images.numpy().ravel(), cmap='viridis', s=50, edgecolor='none')
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=10, edgecolor='none')
    plt.colorbar(label='Input Value (x)')
    plt.title('UMAP Projection of Hidden Layer Activations')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig(f'hidden_layer\\mnist_umap_12d.png')
















