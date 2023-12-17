import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from model import *
from data import load_iris_dataset, load_mnist_dataset
from loss import PerceptualLoss, SSIMLoss


# 损失函数和优化器
def get_optimizer_and_criterion(model, learning_rate, loss_type, use_regularization):
    if use_regularization:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2正则化
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "mae":
        criterion = nn.L1Loss()
    elif loss_type == "huber":
        criterion = nn.SmoothL1Loss()
    elif loss_type == "perceptual":
        criterion = PerceptualLoss()
    elif loss_type == "ssim":
        criterion = SSIMLoss()
    return optimizer, criterion


# 训练函数
def train(model, data_loader, optimizer, criterion, device, l1_lambda=0.001):
    model.train()
    total_loss = 0
    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        if criterion.__class__.__name__ == 'CrossEntropyLoss':
            targets = targets.long()
        elif criterion.__class__.__name__ == 'SSIMLoss':
            outputs = outputs.view(-1, 1, 28, 28)  # 重塑输出
            targets = data  # 使用原始图像作为目标
        else:
            targets = data.view(data.size(0), -1)  # 展平目标，使之与输出形状相匹配
        outputs = outputs.view(-1, 1, 28, 28)  # 重塑输出
        targets = targets.view(-1, 1, 28, 28)
        loss = criterion(outputs, targets)

        # L1 正则化
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += l1_lambda * l1_norm

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


# 测试函数
def test(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            if criterion.__class__.__name__ == 'CrossEntropyLoss':
                targets = targets.long()
            elif criterion.__class__.__name__ == 'SSIMLoss':
                outputs = outputs.view(-1, 1, 28, 28)  # 重塑输出
                targets = data  # 使用原始图像作为目标
            else:
                targets = data.view(data.size(0), -1)  # 展平目标，使之与输出形状相匹配
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)


def test_and_evaluate(model, data_loader, criterion, device, num_images_to_display=10):
    model.eval()
    total_loss = 0
    original_images = []
    reconstructed_images = []

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            outputs = model(data)

            if criterion.__class__.__name__ == 'SSIMLoss':
                outputs = outputs.view(-1, 1, 28, 28)  # 重塑输出
            else:
                outputs = outputs.view(data.size(0), -1)  # 对于其他损失函数，展平输出

            loss = criterion(outputs,
                             data if criterion.__class__.__name__ == 'SSIMLoss' else data.view(data.size(0), -1))
            total_loss += loss.item()

            if len(original_images) < num_images_to_display:
                original_images.extend(data.cpu().numpy())
                reconstructed = outputs.view(outputs.size(0), 1, 28,
                                             28).cpu().numpy() if criterion.__class__.__name__ == 'SSIMLoss' else outputs.cpu().numpy()
                reconstructed_images.extend(reconstructed)

    average_loss = total_loss / len(data_loader)

    # 可视化重建的图像
    fig, axs = plt.subplots(2, num_images_to_display, figsize=(12, 4))
    for i in range(num_images_to_display):
        axs[0, i].imshow(np.squeeze(original_images[i]), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(np.squeeze(reconstructed_images[i]).reshape(28, 28), cmap='gray')
        axs[1, i].axis('off')
    plt.show()

    return average_loss


# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
autoencoder_mnist = ConvolutionalAutoencoder_3()

# 加载数据集
mnist_train_loader, mnist_test_loader = load_mnist_dataset()

# 设置超参数
learning_rate = 0.001
epochs = 3
loss_type = "mse"  # 由于是自编码器，通常使用MSE
use_regularization = True  # 或者设置为True

# 初始化优化器和损失函数
optimizer, criterion = get_optimizer_and_criterion(autoencoder_mnist, learning_rate, loss_type, use_regularization)

# 训练模型
for epoch in range(epochs):
    train_loss = train(autoencoder_mnist, mnist_train_loader, optimizer, criterion, device)
    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}')

# 测试模型
test_loss = test_and_evaluate(autoencoder_mnist, mnist_test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}')
