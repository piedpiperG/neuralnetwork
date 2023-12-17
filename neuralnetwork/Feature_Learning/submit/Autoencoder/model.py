import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
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
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入数据
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class EnhancedAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(EnhancedAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),  # 增加神经元数量
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # 增加了一个额外的层
            nn.ReLU(),
            nn.Linear(32, 3)  # 压缩到3个特征
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),  # 增加神经元数量
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入数据
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AdvancedAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(AdvancedAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 3)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入数据
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class OptimizedAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(OptimizedAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 3)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入数据
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 输入通道1, 输出通道16, 核大小3
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 输入通道16, 输出通道32, 核大小3
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # 输入通道32, 输出通道64, 核大小7
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # 输入通道64, 输出通道32, 核大小7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 输入通道32, 输出通道16, 核大小3
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 输入通道16, 输出通道1, 核大小3
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvolutionalAutoencoder_2(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder_2, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvolutionalAutoencoder_3(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder_3, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=2, dilation=2), # 空洞卷积
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=2, output_padding=1, dilation=2), # 空洞卷积
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x