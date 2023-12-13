import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的全连接自动编码器
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)  # 压缩到3个特征
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # 输出像素值在0到1之间
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 实例化模型、损失函数和优化器
autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 设置训练参数
epochs = 5
batch_size = 64
learning_rate = 1e-3

# 数据集转换器：将数据转换为Tensor并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义训练函数
def train(model, criterion, optimizer, data_loader):
    model.train()
    total_loss = 0
    for data in data_loader:
        inputs, _ = data
        inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


# 定义测试函数
def test(model, criterion, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, _ = data
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    return total_loss / len(data_loader)


# 训练和测试模型
train_losses = []
test_losses = []

for epoch in range(epochs):
    train_loss = train(autoencoder, criterion, optimizer, train_loader)
    test_loss = test(autoencoder, criterion, test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}')

# 返回训练和测试损失
print(train_losses)
print(test_losses)
