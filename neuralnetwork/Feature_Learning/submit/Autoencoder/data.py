from sklearn.datasets import load_iris
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# 鸢尾花数据集导入的封装
class IrisDataset(Dataset):
    def __init__(self):
        iris = load_iris()
        self.data = iris.data.astype(np.float32)
        self.targets = iris.target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def load_iris_dataset(batch_size=64):
    iris_dataset = IrisDataset()
    iris_loader = DataLoader(iris_dataset, batch_size=batch_size, shuffle=True)
    return iris_loader


# MNIST手写数字数据集导入的封装
def load_mnist_dataset(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 示例：载入鸢尾花数据集和MNIST数据集
# iris_loader = load_iris_dataset()
# mnist_train_loader, mnist_test_loader = load_mnist_dataset()

# 显示数据集信息
# len(iris_loader), len(mnist_train_loader), len(mnist_test_loader)
