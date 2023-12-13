import torch
import torch.nn as nn
import torch.optim as optim
from model import Autoencoder, ConvolutionalAutoencoder
from data import load_iris_dataset, load_mnist_dataset


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
    return optimizer, criterion


# 训练函数
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        if criterion.__class__.__name__ == 'CrossEntropyLoss':
            targets = targets.long()
        else:
            targets = data.view(data.size(0), -1)  # 展平目标，使之与输出形状相匹配
        loss = criterion(outputs, targets)
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
            else:
                targets = data.view(data.size(0), -1)  # 展平目标，使之与输出形状相匹配
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)


# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
autoencoder_mnist = Autoencoder(28*28).to(device)
conv_autoencoder = ConvolutionalAutoencoder().to(device)

# 加载数据集
iris_loader = load_iris_dataset()
mnist_train_loader, mnist_test_loader = load_mnist_dataset()

# 训练和测试循环
learning_rate = 0.001
for model in [autoencoder_mnist]:
    for loss_type in ["mse", "cross_entropy"]:
        for use_regularization in [True, False]:
            for loader in [mnist_train_loader]:

                print(
                    f"Model: {model.__class__.__name__}, Loss: {loss_type}, Regularization: {use_regularization}, "
                    f"loader: {loader}")
                optimizer, criterion = get_optimizer_and_criterion(model, learning_rate, loss_type, use_regularization)
                # 训练模型
                for epoch in range(2):  # 假设训练5个Epoch
                    train_loss = train(model, loader, optimizer, criterion, device)
                    test_loss = test(model, loader, criterion, device)
                    print(
                        f"Model: {model.__class__.__name__}, Loss: {loss_type}, Regularization: {use_regularization}, "
                        f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
