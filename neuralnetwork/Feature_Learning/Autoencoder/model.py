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


