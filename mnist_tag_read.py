
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

# Example usage:
# Assuming you have the files in './data/mnist/' and want to apply transformations like ToTensor()
# from torchvision import transforms


dataset = MNISTTagDataset(
    label_file_path='./mnist/t10k-labels.idx1-ubyte',
)

# You can then use DataLoader to iterate over the dataset.

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
first_batch = next(iter(dataloader))
for i, label in enumerate(first_batch):
    print("Label of image", i, ":", label.item())


