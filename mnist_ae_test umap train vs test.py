import struct
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader



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

# Train set path
train_images_path = 'mnist/train-images.idx3-ubyte'
train_labels_path = 'mnist/train-labels.idx1-ubyte'

# Load Train set
train_dataset = MNISTDataset(train_images_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# 测试集路径
test_images_path = 'mnist/t10k-images.idx3-ubyte'
test_labels_path = 'mnist/t10k-labels.idx1-ubyte'



# 加载测试数据集
test_dataset = MNISTDataset(test_images_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# test_label_dataset = MNISTTagDataset(label_file_path=test_labels_path)
# test_label_loader = DataLoader(test_label_dataset, batch_size=len(test_label_dataset), shuffle=False)


import umap
from mnist_lib import MNISTTagDataset
from matplotlib.colors import ListedColormap
# 测试模型性能
with torch.no_grad():
    correct = 0
    total = 0

    
    all_outputs = []



    for data in train_loader:  # 假设你不使用标签进行预测
        data = data.to(device)
        outputs = model.encoder(data)
        all_outputs.append(outputs.cpu())  # 将输出移到CPU并保存

    for data in test_loader:  # 假设你不使用标签进行预测
        data = data.to(device)
        outputs = model.encoder(data)
        all_outputs.append(outputs.cpu())  # 将输出移到CPU并保存
    final_output = torch.cat(all_outputs, dim=0)

    #  display tensor shape
    print(f"Final Output Shape: {final_output.shape}" )

    # first 60000 ones are from train set, labels to 1.
    # other 10000 ones are labeled to 0.
    colors = np.zeros(len(final_output))
    colors[:len(train_loader.dataset)] = 1


    # Apply UMAP to reduce dimensions to 2D for visualization
    reducer = umap.UMAP(n_components=2, random_state=42)
    # reducer = umap.UMAP(min_dist=0.5)
    embedding = reducer.fit_transform(final_output)
    # print embedding shape
    print("Embedding shape:", embedding.shape)


    # input_cpu = inputs.detach().cpu().numpy()


    # Define a list of 10 distinct colors for categories 0 through 9
    colors_map = [
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

    colors_map = [
        '#d62728',  # brick red

        '#7f7f7f',  # middle gray
    ]

    # Create a ListedColormap from the list of colors
    cmap = ListedColormap(colors_map)

    figsize = (10, 7)


    # Plot the embeddings
    plt.figure(figsize=figsize)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=input_images.numpy().ravel(), cmap='viridis', s=50, edgecolor='none')
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap='viridis', s=10, edgecolor='none')
    plt.colorbar(label='Input Value (x)')
    plt.title('UMAP Projection of Hidden Layer Activations')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig(f'hidden_layer\\mnist_umap_12d_test_vs_train.png')
















