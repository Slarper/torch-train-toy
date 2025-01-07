import struct

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
            yield image  # 使用生成器逐个返回图像

import numpy as np
import matplotlib.pyplot as plt

# 假设我们已经有一个 image 的迭代器
image_iterator = read_images('mnist/train-images.idx3-ubyte')

# 获取第一张图片并转换成 28x28 的矩阵
image = next(image_iterator)
image_matrix = np.array(image).reshape((28, 28))

# 显示图像
plt.imshow(image_matrix, cmap='gray')
# plt.show()
plt.savefig('mnist_output/first_image.png')  # 保存图像到文件

for i in range(10):  # 获取并显示前10张图片
    image = next(image_iterator)
    image_matrix = np.array(image).reshape((28, 28))
    plt.imshow(image_matrix, cmap='gray')
    plt.savefig(f'mnist_output/image_{i}.png')  # 保存图像到文件

def show_images(images):
    for i, image in enumerate(images):
        image_matrix = np.array(image).reshape((28, 28))
        plt.imshow(image_matrix, cmap='gray')
        plt.savefig(f'mnist_output/image_{i}.png')  # 保存图像到文件

