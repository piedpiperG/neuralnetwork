import torch
import torchvision.models as models

# 创建VGG19模型实例
vgg19 = models.vgg19(pretrained=True)

# 将模型设为评估模式
vgg19.eval()

# 遍历features模块计算卷积和池化层的神经元数量
neurons = 0
input_tensor = torch.rand(1, 3, 224, 224)  # 假设的输入，1x3x224x224
current_shape = input_tensor.shape[1:]  # 初始shape忽略批处理维度
cnt_conv = 0
cnt_pool = 0
cnt_link = 0

for layer in vgg19.features:
    input_tensor = layer(input_tensor)
    # 仅计算卷积层和池化层的神经元数量
    if isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
        # 特征图的尺寸: 通道数 x 宽度 x 高度
        current_shape = input_tensor.shape[1:]
        neurons += current_shape[0] * current_shape[1] * current_shape[2]
        if isinstance(layer, torch.nn.Conv2d):
            cnt_conv += 1
            print(f'conv{cnt_conv}:{current_shape[0]} * {current_shape[1]} * {current_shape[2]}')
        if isinstance(layer, torch.nn.MaxPool2d):
            cnt_pool += 1
            print(f'pool{cnt_pool}:{current_shape[0]} * {current_shape[1]} * {current_shape[2]}')


# 遍历classifier模块计算全连接层的神经元数量
for layer in vgg19.classifier:
    if isinstance(layer, torch.nn.Linear):
        neurons += layer.out_features
        cnt_link += 1
        print(f'link{cnt_link}:{layer.out_features}')

print("Theoretical Neurons:", neurons)

# 计算可训练参数的数量
parameters = sum(p.numel() for p in vgg19.parameters() if p.requires_grad)
print("Theoretical Parameters:", parameters)
