import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
from torchvision.datasets import CIFAR10
import torch.nn.init as init
import matplotlib.pyplot as plt
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_size, n_g_feature, n_channel, num_classes):
        super(ConditionalGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_size)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size * 2, 4 * n_g_feature, kernel_size=4, bias=False),
            nn.BatchNorm2d(4 * n_g_feature),
            nn.ReLU(),

            nn.ConvTranspose2d(4 * n_g_feature, 2 * n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_g_feature),
            nn.ReLU(),

            nn.ConvTranspose2d(2 * n_g_feature, n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_g_feature),
            nn.ReLU(),

            nn.ConvTranspose2d(n_g_feature, n_channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c.unsqueeze(2).unsqueeze(3)], 1)
        return self.main(x)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, n_channel, n_d_feature, num_classes):
        super(ConditionalDiscriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, n_channel * 32 * 32)

        self.main = nn.Sequential(
            nn.Conv2d(n_channel * 2, n_d_feature, kernel_size=4, stride=2, padding=1)
            ,
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * n_d_feature),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4 * n_d_feature, 1, kernel_size=4)
        )

    def forward(self, x, labels):
        c = self.label_emb(labels).view(-1, n_channel, 32, 32)
        x = torch.cat([x, c], 1)
        return self.main(x)


dataset = CIFAR10(root='./CIFARdata', download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# 模型参数
latent_size = 1024  # 假设潜在空间维度为100
n_channel = 3  # 图像通道数
n_g_feature = 64  # 生成器特征数
n_d_feature = 64  # 判别器特征数
num_classes = 1  # CIFAR10的类别数

# 创建生成器和判别器实例
netG = ConditionalGenerator(latent_size, n_g_feature, n_channel, num_classes).to(device)
netD = ConditionalDiscriminator(n_channel, n_d_feature, num_classes).to(device)


def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)


# 应用权重初始化
netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCEWithLogitsLoss()

# 为生成器和判别器设置优化器
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.00000002, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.00000002, betas=(0.5, 0.999))

# 参数设置
batch_size = 64
fixed_noises = torch.randn(batch_size, latent_size, 1, 1)
fixed_labels = torch.randint(0, num_classes, (batch_size,))

# 初始化用于跟踪损失的列表
d_losses = []
g_losses = []

# 训练循环
epoch_num = 3
for epoch in range(epoch_num):
    for batch_idx, (real_images, real_labels) in enumerate(dataloader):
        current_batch_size = real_images.size(0)
        real_labels = torch.ones(current_batch_size, 1).to(device)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)

        # 为每个批次生成随机标签
        labels = torch.randint(0, num_classes, (current_batch_size,), device=device)

        # 训练判别器
        netD.zero_grad()
        real_images = real_images.to(device)
        labels = labels.to(device)
        outputs = netD(real_images, labels).view(-1, 1)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()
        D_x = outputs.sigmoid().mean()

        # 生成假图像
        noise = torch.randn(current_batch_size, latent_size, 1, 1, device=device)
        fake_images = netG(noise, labels)
        outputs = netD(fake_images.detach(), labels).view(-1, 1)
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        D_G_z1 = outputs.sigmoid().mean()

        # 更新判别器
        d_loss = d_loss_real + d_loss_fake
        optimizerD.step()

        # 训练生成器
        netG.zero_grad()
        outputs = netD(fake_images.detach(), labels).view(-1, 1)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        D_G_z2 = outputs.sigmoid().mean()
        optimizerG.step()

        # 记录损失
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        if batch_idx % 100 == 0:
            print(
                f'Epoch [{epoch}/{epoch_num}] Batch {batch_idx}/{len(dataloader)} Loss D: {d_loss:.4f}, Loss G: {g_loss:.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            # 保存图像
            with torch.no_grad():
                fake = netG(fixed_noises.to(device), fixed_labels.to(device)).detach().to(device)
                save_image(fake, f'./CGAN_saved/images_epoch{epoch:02d}_batch{batch_idx:03d}.png')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses, label="Generator")
plt.plot(d_losses, label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 保存生成器和判别器
generator_save_path = 'cgan_generator.pt'
torch.save(netG.state_dict(), generator_save_path)

discriminator_save_path = 'cgan_discriminator.pt'
torch.save(netD.state_dict(), discriminator_save_path)

# 重新创建模型实例
netG = ConditionalGenerator(latent_size, n_g_feature, n_channel, num_classes)
netD = ConditionalDiscriminator(n_channel, n_d_feature, num_classes)

# 加载保存的状态字典
netG.load_state_dict(torch.load(generator_save_path))
netD.load_state_dict(torch.load(discriminator_save_path))

# 将模型设置为评估模式
netG.eval()
netD.eval()

# 生成图像以评估模型
with torch.no_grad():
    for i in range(100):
        # 生成随机噪声和随机标签
        noises = torch.randn(batch_size, latent_size, 1, 1)
        random_labels = torch.randint(0, num_classes, (batch_size,))

        # 生成图像
        fake_images = netG(noises, random_labels).detach().to(device)

        # 保存生成的图像
        save_image(fake_images, f'./CGAN_Generated_Images/{i}.png')
