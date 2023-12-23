import torch.nn as nn
import torch.nn.init as init
import torch
import torch.optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

latent_size = 64
n_channel = 3
n_g_feature = 64
n_d_feature = 64
gnet = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4, bias=False),
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

dnet = nn.Sequential(
    nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * n_d_feature),
    nn.LeakyReLU(0.2),

    nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(4 * n_d_feature),
    nn.LeakyReLU(0.2),

    nn.Conv2d(4 * n_d_feature, 1, kernel_size=4)
)


def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)


gnet.apply(weights_init)
dnet.apply(weights_init)

dataset = CIFAR10(root='./CIFARdata', download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
goptimizer = torch.optim.Adam(gnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
doptimizer = torch.optim.Adam(dnet.parameters(), lr=0.0002, betas=(0.5, 0.999))

batch_size = 64
fixed_noises = torch.randn(batch_size, latent_size, 1, 1)

# 初始化用于跟踪损失的列表
d_losses = []
g_losses = []

epoch_num = 1
for epoch in range(epoch_num):
    for batch_idx, data in enumerate(dataloader):
        real_images, _ = data
        batch_size = real_images.size(0)

        # 首先，用真实图像训练判别器
        labels = torch.ones(batch_size)
        preds = dnet(real_images)
        outputs = preds.reshape(-1)
        dloss_real = criterion(outputs, labels)
        dmean_real = outputs.sigmoid().mean()

        # 接着，用假图像训练判别器
        noises = torch.randn(batch_size, latent_size, 1, 1)
        fake_images = gnet(noises)
        labels = torch.zeros(batch_size)
        fake = fake_images.detach()

        preds = dnet(fake)
        outputs = preds.view(-1)
        dloss_fake = criterion(outputs, labels)
        dmean_fake = outputs.sigmoid().mean()

        # 更新判别器
        dloss = dloss_real + dloss_fake
        dnet.zero_grad()
        dloss.backward()
        doptimizer.step()

        # 训练生成器
        labels = torch.ones(batch_size)
        preds = dnet(fake_images)
        outputs = preds.view(-1)
        gloss = criterion(outputs, labels)
        gmean_fake = outputs.sigmoid().mean()
        gnet.zero_grad()
        gloss.backward()
        goptimizer.step()

        # 记录损失
        d_losses.append(dloss.item())
        g_losses.append(gloss.item())

        if batch_idx % 100 == 0:
            fake = gnet(fixed_noises)
            save_image(fake, f'./GAN_saved02/images_epoch{epoch:02d}_batch{batch_idx:03d}.png')

            print(f'Epoch index: {epoch}, {epoch_num} epoches in total.')
            print(f'Batch index: {batch_idx}, the batch size is {batch_size}.')
            print(f'Discriminator loss is: {dloss}, generator loss is: {gloss}', '\n',
                  f'Discriminator tells real images real ability: {dmean_real}', '\n',
                  f'Discriminator tells fake images real ability: {dmean_fake:g}/{gmean_fake:g}')
# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.title("训练过程中生成器和判别器的损失")
plt.plot(g_losses, label="生成器")
plt.plot(d_losses, label="判别器")
plt.xlabel("迭代次数")
plt.ylabel("损失")
plt.legend()
plt.show()

gnet_save_path = 'gnet.pt'
torch.save(gnet, gnet_save_path)

dnet_save_path = 'dnet.pt'
torch.save(dnet, dnet_save_path)

gnet = torch.load(gnet_save_path)
gnet.eval()

dnet = torch.load(dnet_save_path)
dnet.eval()

for i in range(100):
    noises = torch.randn(batch_size, latent_size, 1, 1)
    fake_images = gnet(noises)
    save_image(fake, f'./G_Data/{i}.png')
