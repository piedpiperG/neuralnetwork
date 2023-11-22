import torch
import torchvision.models as models

# 创建一个ResNet152网络的实例
resnet152 = models.resnet152(pretrained=True)

# 计算神经元数目
neurons = 0


# 钩子函数，用于计算每个层的输出大小
def hook_function(module, input, output):
    global neurons
    neurons += output.numel()


# 注册钩子
for layer in resnet152.modules():
    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
        layer.register_forward_hook(hook_function)

# 假设输入，1x3x224x224
input_tensor = torch.rand(1, 3, 224, 224)

# 前向传播，计算神经元数量
output = resnet152(input_tensor)

# 打印神经元数目
print("Neurons:", neurons)

# 计算可训练参数的数量
parameters_count = sum(p.numel() for p in resnet152.parameters() if p.requires_grad)
print("Parameters:", parameters_count)
